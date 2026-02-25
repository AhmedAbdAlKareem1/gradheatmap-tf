import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatMapPyTorch:
    def __init__(
        self,
        model,
        img_path,
        class_names=None,
        target_class=None,
        preprocess=None,
        image_size=None,
        device=None,
    ):
        """
        model:
            - torch.nn.Module OR
            - str path to a checkpoint (.pth/.pt) that contains model_state_dict
              (you must also provide a model_builder if you want auto-load; see load_checkpoint()).
        preprocess:
            - None -> auto: if preprocess is string or model_hint provided, use ImageNet normalization
            - callable -> your custom preprocessing
            - str -> one of: 'resnet', 'resnet50', 'vgg16', 'mobilenet', 'efficientnet', ...
        image_size:
            - None -> default 224
            - tuple (H, W)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If model is a path, user should load it before passing, or use load_checkpoint helper below.
        if isinstance(model, str):
            raise ValueError(
                "Pass a torch.nn.Module (already built). "
                "If you have a .pth file, load it into a model first, then pass the model object."
            )

        self.model = model.to(self.device).eval()

        self.img_path = img_path
        self.class_names = class_names if class_names is not None else []
        self.target_class = target_class

        self.user_preprocess = preprocess
        self.image_size = image_size if image_size is not None else (224, 224)

        # Hooks buffers
        self.gradients = None
        self.activations = None

        # Auto-select layer
        self.target_layer, self.target_layer_name = self.find_last_conv_layer()
        print(f"Targeting Layer: {self.target_layer_name}")
        print(f"Image Size: {self.image_size}")

    # ---------------------------
    # Layer detection
    # ---------------------------
    def find_last_conv_layer(self):
        last_conv = None
        last_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                last_name = name
        if last_conv is None:
            raise ValueError("No Conv2d layer found in the model. Grad-CAM requires a conv layer.")
        return last_conv, last_name

    # ---------------------------
    # Hooks
    # ---------------------------
    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; grad_output[0] matches activation gradient
        self.gradients = grad_output[0]

    # ---------------------------
    # Preprocessing helpers
    # ---------------------------
    def _imagenet_norm(self, x):
        # x: float tensor [1,3,H,W] in range [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def get_preprocess_function(self):
        """
        Similar idea to your TF mapping.
        Here we mainly decide if we apply ImageNet normalization or not.
        """
        mapping = {
            "vgg16": "imagenet",
            "vgg19": "imagenet",
            "resnet": "imagenet",
            "resnet50": "imagenet",
            "resnet101": "imagenet",
            "resnet152": "imagenet",
            "mobilenet": "imagenet",
            "mobilenetv2": "imagenet",
            "mobilenetv3": "imagenet",
            "efficientnet": "imagenet",
            "efficientnetv2": "imagenet",
            "densenet": "imagenet",
            "inception": "imagenet",
            "xception": "imagenet",
            "convnext": "imagenet",
            "regnet": "imagenet",
        }

        p = self.user_preprocess

        # 1) user callable
        if callable(p):
            print("Using USER preprocessing function.")
            return p

        # 2) user string
        if isinstance(p, str):
            key = p.lower().strip()
            if key not in mapping:
                raise ValueError(f"Unknown preprocess='{p}'. Valid keys: {sorted(mapping.keys())}")
            print(f"Using USER preprocess='{p}' (ImageNet normalization).")
            return "imagenet"

        # 3) default: None (no special preprocess)
        return None

    def preprocess_image(self):
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f"File not found: {self.img_path}")

        img_bgr = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError(f"cv2.imread failed: {self.img_path}")

        # Detect model input channels from first conv
        first_conv = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                first_conv = m
                break
        in_channels = first_conv.in_channels if first_conv is not None else 3

        if in_channels == 1:
            # grayscale path
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (self.image_size[1], self.image_size[0]))
            x = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]
            return x.to(self.device)

        # RGB path
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.image_size[1], self.image_size[0]))

        # User callable preprocess gets raw uint8 RGB image (H,W,3)
        p = self.get_preprocess_function()
        if callable(p):
            out = p(img_rgb)
            if not isinstance(out, torch.Tensor):
                raise ValueError("Your preprocess callable must return a torch.Tensor.")
            if out.ndim != 4:
                raise ValueError("Your preprocess callable must return shape [1,C,H,W].")
            return out.to(self.device)

        # Default tensor conversion
        x = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # [1,3,H,W]
        x = x.to(self.device)

        # ImageNet normalization if requested
        if p == "imagenet":
            x = self._imagenet_norm(x)

        return x

    # ---------------------------
    # Prediction helper
    # ---------------------------
    def predict(self, x):
        # IMPORTANT: do NOT use torch.no_grad here
        logits = self.model(x)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        # For printing only (detach)
        if logits.shape[-1] == 1:
            prob_pos = torch.sigmoid(logits.detach()).view(-1)[0].item()
            idx = 1 if prob_pos >= 0.5 else 0
            name = self.class_names[idx] if len(self.class_names) >= 2 else str(idx)
            conf = prob_pos if idx == 1 else (1.0 - prob_pos)
            probs = {
                (self.class_names[0] if len(self.class_names) >= 1 else "0"): round(1.0 - prob_pos, 4),
                (self.class_names[1] if len(self.class_names) >= 2 else "1"): round(prob_pos, 4),
            }
            print(f"Class: {idx} {name}  Confidence: {conf * 100:.2f}%")
            return idx, name, conf, probs, logits

        probs_t = torch.softmax(logits.detach(), dim=1)[0]
        idx = int(torch.argmax(probs_t).item())
        conf = float(probs_t[idx].item())
        name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
        probs = {
            (self.class_names[i] if i < len(self.class_names) else str(i)): round(float(probs_t[i].item()), 4)
            for i in range(probs_t.shape[0])
        }
        print(f"Class: {idx} ({name})")
        print(f"Confidence: {conf * 100:.2f}%")
        return idx, name, conf, probs, logits

    # ---------------------------
    # Grad-CAM
    # ---------------------------
    def compute_gradcam(self):
        x = self.preprocess_image()
        x.requires_grad_(True)

        # Register hooks
        f_hook = self.target_layer.register_forward_hook(self._forward_hook)
        # use full backward hook (newer torch), fallback to backward_hook if needed
        try:
            b_hook = self.target_layer.register_full_backward_hook(self._backward_hook)
        except Exception:
            b_hook = self.target_layer.register_backward_hook(self._backward_hook)

        try:
            idx, name, conf, probs, logits = self.predict(x)

            # Decide which class to explain
            if logits.shape[-1] == 1:
                # binary logit
                explain_class = int(self.target_class) if self.target_class is not None else int(idx)
                # If explain_class==1 -> use logit; if 0 -> use negative logit
                score = logits[:, 0] if explain_class == 1 else (-logits[:, 0])
            else:
                class_index = int(self.target_class) if self.target_class is not None else int(torch.argmax(logits, dim=1).item())
                score = logits[:, class_index]

            self.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)

            if self.gradients is None or self.activations is None:
                raise ValueError(
                    "Gradients/activations are None. "
                    "The targeted layer might be disconnected from the output."
                )

            # activations: [B,C,H,W]
            # gradients:   [B,C,H,W]
            grads = self.gradients
            acts = self.activations

            # Global average pool gradients over spatial dims -> [C]
            pooled_grads = torch.mean(grads, dim=(0, 2, 3))

            # Weight channels
            weighted = acts.detach().clone()
            for c in range(weighted.shape[1]):
                weighted[:, c, :, :] *= pooled_grads[c]

            # Heatmap: mean over channels -> [H,W]
            heatmap = torch.mean(weighted, dim=1).squeeze(0)
            heatmap = F.relu(heatmap)

            # Normalize to [0,1]
            maxv = torch.max(heatmap)
            if maxv > 0:
                heatmap = heatmap / maxv

            return heatmap.detach().cpu().numpy()

        finally:
            f_hook.remove()
            b_hook.remove()

    def overlay_heatmap(self, alpha=0.4):
        heatmap = self.compute_gradcam()
        if heatmap is None or not isinstance(heatmap, np.ndarray) or heatmap.size == 0:
            raise ValueError("Grad-CAM heatmap is empty.")

        img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError(f"cv2.imread failed: {self.img_path}")

        h, w = img.shape[:2]

        heatmap = heatmap.astype(np.float32)
        if heatmap.ndim != 2:
            raise ValueError(f"Heatmap must be 2D, got: {heatmap.shape}")

        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        out = cv2.addWeighted(img, alpha, heatmap_color, 1 - alpha, 0)
        if out is None or out.size == 0:
            raise ValueError("Overlay result is empty.")
        return out

    def save_heat_img(self, name, output_img):
        folder = "heatmap"
        os.makedirs(folder, exist_ok=True)

        if output_img is None or (isinstance(output_img, np.ndarray) and output_img.size == 0):
            raise ValueError("save_heat_img got empty output_img.")

        # Add default extension if missing
        root, ext = os.path.splitext(name)
        if ext.strip() == "":
            name = name + ".jpg"

        save_path = os.path.join(folder, name)

        ok = cv2.imwrite(save_path, output_img)
        if not ok:
            raise IOError(f"cv2.imwrite failed: {save_path}")

        print(f"Saved: {save_path}")