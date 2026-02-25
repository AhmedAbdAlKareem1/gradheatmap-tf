from setuptools import setup, find_packages

setup(
    name="gradheatmap",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
    ],
    extras_require={
        "tf": ["tensorflow>=2.0"],
        "torch": ["torch>=1.0", "torchvision"],
    },
    author="Ahmed Abd AlKareem",
    description="Automatic Grad-CAM heatmaps for TensorFlow and PyTorch",
    license="MIT",
    python_requires=">=3.8",
)
