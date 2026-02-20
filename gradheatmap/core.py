import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2

class HeatMap:
    def __init__(self, model, img_path,class_names,target_class=None):
        self.model = load_model(model, compile=False)
        self.pre_trained_model = self.detect_backbone_submodel()
        if self.pre_trained_model is None:
            self.pre_trained_model = self.model
            self.backbone_name = None
        else:
            self.backbone_name = self.pre_trained_model.name
        self.img_path = img_path
        self.image_size = (self.model.input_shape[1:3])
        print(f"Image Size = {self.image_size}")
        self.class_names = class_names
        self.target_class = target_class

    def overlay_heatmap(self, alpha=0.4):
        try:
            heatmap = self.compute_gradcam()
            if heatmap is None or not isinstance(heatmap, np.ndarray) or heatmap.size == 0:
                raise ValueError("Grad-CAM heatmap is empty (compute_gradcam returned None/empty).")

            img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise ValueError(f"cv2.imread failed. Bad path or unreadable image: {self.img_path}")

            # Ensure heatmap is float32 and 2D
            heatmap = np.asarray(heatmap)
            if heatmap.ndim != 2:
                raise ValueError(f"Heatmap must be 2D, got shape: {heatmap.shape}")

            heatmap = heatmap.astype(np.float32)

            h, w = img.shape[:2]
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

            # Normalize to [0, 255] safely
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = heatmap.astype(np.uint8)

            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Blend (img is BGR, heatmap_color is BGR -> OK)
            superimposed = cv2.addWeighted(img, alpha, heatmap_color, 1 - alpha, 0)

            if superimposed is None or superimposed.size == 0:
                raise ValueError("Overlay result is empty.")

            return superimposed
        except Exception as E:
            print(E)

    def compute_gradcam(self):
        try:
            preprocessed_image = self.preprocess_image()
            inner_layers = self.pre_trained_model
            conv_layer_name = self.last_conv_layer_get(inner_layers)

            # Build the gradient model
            grad_model = tf.keras.models.Model(
                inputs=[inner_layers.input],
                outputs=[inner_layers.get_layer(conv_layer_name).output, inner_layers.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, last_layer_features = grad_model(preprocessed_image)

                # --- FIXED LOGIC START ---
                x = last_layer_features
                start_chaining = False

                if self.backbone_name is None:
                    # Custom Model Path: Start from conv_outputs
                    x = conv_outputs
                    for layer in self.model.layers:
                        if start_chaining:
                            if layer == self.model.layers[-1] and hasattr(layer, 'kernel'):
                                x = tf.matmul(tf.cast(x, layer.kernel.dtype), layer.kernel) + layer.bias
                            else:
                                x = layer(x)

                        if layer.name == conv_layer_name:
                            start_chaining = True
                else:
                    # Backbone Path (MobileNet/ResNet): Start from features
                    for layer in self.model.layers:
                        if layer.name == self.backbone_name:
                            start_chaining = True
                            continue  # Skip the backbone layer itself as we already have its output

                        if start_chaining:
                            if layer == self.model.layers[-1] and hasattr(layer, 'kernel'):
                                x = tf.matmul(tf.cast(x, layer.kernel.dtype), layer.kernel) + layer.bias
                            else:
                                x = layer(x)
                # --- FIXED LOGIC END ---
                pred_class = self.predict(preprocessed_image)
                # Loss calculation
                if x.shape[-1] == 1:
                    # Binary sigmoid output
                    if self.target_class is not None:
                        explain_class = int(self.target_class)
                    else:
                        explain_class = int(pred_class)

                    # explain_class=1 -> use x, explain_class=0 -> use (1 - x)
                    loss = x[:, 0] if explain_class == 1 else (1.0 - x[:, 0])
                else:
                    if self.target_class is None:
                        class_index = tf.argmax(x[0])
                    else:
                        class_index = int(self.target_class)
                    loss = x[:, class_index]

            # 5. Calculate gradients with respect to the conv_outputs
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs.numpy()[0]
            pooled_grads = pooled_grads.numpy()

            for i in range(pooled_grads.shape[-1]):
                conv_outputs[:, :, i] *= pooled_grads[i]

            heatmap = np.mean(conv_outputs, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap) + 1e-10)

            return heatmap
        except Exception as E:
            print(f"Error in compute_gradcam: {E}")
            return None

    def get_preprocess_function(self):
        name = self.backbone_name.lower()

        # VGG
        if "vgg16" in name:
            return tf.keras.applications.vgg16.preprocess_input
        if "vgg19" in name:
            return tf.keras.applications.vgg19.preprocess_input

        # ResNet
        if "resnet" in name and "v2" not in name:
            return tf.keras.applications.resnet.preprocess_input
        if "resnet" in name and "v2" in name:
            return tf.keras.applications.resnet_v2.preprocess_input

        # MobileNet
        if "mobilenetv3" in name:
            return tf.keras.applications.mobilenet_v3.preprocess_input
        if "mobilenetv2" in name:
            return tf.keras.applications.mobilenet_v2.preprocess_input
        if "mobilenet" in name:
            return tf.keras.applications.mobilenet.preprocess_input

        # EfficientNet
        if "efficientnetv2" in name:
            return tf.keras.applications.efficientnet_v2.preprocess_input
        if "efficientnet" in name:
            return tf.keras.applications.efficientnet.preprocess_input

        # DenseNet
        if "densenet" in name:
            return tf.keras.applications.densenet.preprocess_input

        # Inception
        if "inceptionresnet" in name:
            return tf.keras.applications.inception_resnet_v2.preprocess_input
        if "inception" in name:
            return tf.keras.applications.inception_v3.preprocess_input

        # Xception
        if "xception" in name:
            return tf.keras.applications.xception.preprocess_input

        # NASNet
        if "nasnet" in name:
            return tf.keras.applications.nasnet.preprocess_input

        # ConvNeXt
        if "convnext" in name:
            return tf.keras.applications.convnext.preprocess_input

        # RegNet
        if "regnet" in name:
            return tf.keras.applications.regnet.preprocess_input

        return None
    def preprocess_image(self):
        try:

            if not os.path.exists(self.img_path):
                print(f"Error : File Not Found {self.img_path}")
                raise FileNotFoundError
            img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            img = np.expand_dims(img, axis=0)  # (1,img_size,img_size,3)
            #1 is the image, and 3 is the channel (RGB)
            # 2. Check for Backbone Scaling
            if self.backbone_name:
                process_func = self.get_preprocess_function()
                if process_func:
                    print(f"Using {self.backbone_name} specific preprocessing.")
                    return process_func(img)

            # 3. Detect Internal Rescaling Layer
            has_internal_rescaling = False
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Rescaling):
                    has_internal_rescaling = True
                    print(f"Internal scaling detected ({layer.scale}). Passing raw pixels.")
                    print("  offset:", layer.offset)
                    break

            # 4. Final Decision
            if has_internal_rescaling:
                return img  # Model will divide by 255 internally
            else:
                print("No internal scaling or backbone found. Manually scaling to [0, 1].")
                return img / 255.0  # Apply manual scaling

        except Exception as E:
            print(f"Preprocess Error: {E}")

    def last_conv_layer_get(self,model):
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                print(f"Layer Name : {layer.name}")
                return layer.name
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        print(f"Last Convo Layer : {sub_layer.name}")
                        return sub_layer.name
        return None

    def predict(self, img):
        try:
            preds = self.model(img, training=False).numpy()[0]

            if preds.shape[-1] == 1:
                # Binary case
                score = float(preds[0])
                class_index = 1 if score >= 0.5 else 0
                confidence = score if class_index == 1 else (1.0 - score)
                name = self.class_names[class_index] if self.class_names else str(class_index)
                print(f"Class: {class_index} {name}  Confidence: {confidence * 100:.2f}%")
                return class_index

            # Multi-class case
            class_index = int(np.argmax(preds))
            confidence = float(preds[class_index])
            name = self.class_names[class_index] if self.class_names else str(class_index)

            print(f"Class: {class_index} ({name})")
            print(f"Confidence: {confidence * 100:.2f}%")
            return class_index

        except Exception as e:
            print(f"Predict Error: {e}")
            return 0

    def detect_backbone_submodel(self):
        # Candidates are nested Models inside the top model
        candidates = [l for l in self.model.layers if isinstance(l, tf.keras.Model)]

        if not candidates:
            print(f"No nested backbone. Using top model name: {self.model.name}")
            return self.model  # treat full model as backbone

        # Heuristic: pick the nested model with the most parameters
        # (backbones usually dominate params, head is small)
        backbone = max(candidates, key=lambda m: m.count_params())
        print(f"Detected Model : {backbone.name}")
        return backbone
    def save_heat_img(self, name, output_img):
        try:
            folder_name = 'heatmap'
            os.makedirs(folder_name, exist_ok=True)

            if output_img is None or (isinstance(output_img, np.ndarray) and output_img.size == 0):
                raise ValueError("save_heat_img got an empty output_img (None/empty).")

            save_path = os.path.join(folder_name, name)
            if output_img is None:
                return Exception
            ok = cv2.imwrite(save_path, output_img)
            tf.keras.backend.clear_session()

            if not ok:
                raise IOError(f"cv2.imwrite failed for path: {save_path}")

            print(f"Successfully saved heatmap to: {save_path}")
        except Exception as E:
            print(E)


