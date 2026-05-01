# NNDL_TUMOR: Brain Tumor Classification & Explainable AI

This repository contains a high-performance deep learning pipeline for multi-class brain tumor classification from MRI images into four classes: `glioma`, `meningioma`, `notumor`, and `pituitary`.

## 📊 1. Performance Benchmark Results

| Model Architecture | Test Accuracy | Parameters | Model Size | Latency | Inference Speed (FPS) |
|---|---|---|---|---|---|
| **MobileNetV2** | **95.31%** 🏆 | **2.23 Million** | **8.74 MB** | **2.11 ms** | **474.1 FPS** |
| **EfficientNet-B0** | 95.12% | 4.01 Million | 15.59 MB | 3.20 ms | 312.3 FPS |
| **InceptionV3** | 94.19% | 25.12 Million | 96.16 MB | 6.07 ms | 164.7 FPS |
| **ResNet50** | 92.12% | 23.52 Million | 90.00 MB | 2.48 ms | 403.9 FPS |
| **Custom CNN** | 83.75% | 4.85 Million | 18.53 MB | 1.35 ms | 738.8 FPS |

---

## 🛠️ 2. Key Regularization & Training Mechanics

To narrow the accuracy gap between training and validation, the following hyperparameter settings were used:

- **Optimizer**: Adam with model-specific learning rates.
- **Learning Rate Schedule**: Cosine Annealing decay down to $1\times10^{-7}$.
- **Regularization**: Label Smoothing (0.05), weight decay, and dropout are applied to control overfitting.
- **Data Augmentation**: Flips, rotations, Gaussian Blur, and random erasing to improve robustness.

---

## 📂 3. Repository Structure

```
.
├── train.py                  # Core training engine
├── run_all.sh                # Automation script for multi-model runs
├── gradcam.py                # Visualizes individual activations using Grad-CAM
├── gradcam_folder.py         # Generates folder-wise Grad-CAM heatmaps
├── generate_summary_table.py # Compiles Markdown and LaTeX summary tables
├── results/                  # Accuracy curves, confusion matrices, and metrics
└── data/                     # Training & testing MRI scans (ignored by git)
```

---

## 🚀 4. Usage Instructions

To replicate results, configure your environment and run the following commands sequentially:

```bash
# 1. Install required packages
pip install torch torchvision matplotlib seaborn scikit-learn opencv-python grad-cam

# 2. Run the training automation script
bash run_all.sh

# 3. Print benchmarking and performance metrics
python generate_summary_table.py

# 4. Generate the explainable AI heatmaps
python gradcam_folder.py
```
