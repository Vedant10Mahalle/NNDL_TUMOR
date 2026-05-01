# NNDL_TUMOR: Advanced Brain Tumor Classification & Explainable AI

This repository contains a high-performance, production-ready, and explainable deep learning pipeline for multi-class brain tumor classification from MRI images (`glioma`, `meningioma`, `notumor`, `pituitary`).

## 📊 Performance Benchmark Comparison

| Model Architecture | Test Accuracy | Parameters | Model Size | Latency | Inference Speed (FPS) |
|---|---|---|---|---|---|
| **MobileNetV2** | **95.31%** 🏆 | **2.23 Million** | **8.74 MB** | **2.11 ms** | **474.1 FPS** |
| **EfficientNet-B0** | 95.12% | 4.01 Million | 15.59 MB | 3.20 ms | 312.3 FPS |
| **InceptionV3** | 94.19% | 25.12 Million | 96.16 MB | 6.07 ms | 164.7 FPS |
| **ResNet50** | 92.12% | 23.52 Million | 90.00 MB | 2.48 ms | 403.9 FPS |
| **Custom CNN** | 83.75% | 4.85 Million | 18.53 MB | 1.35 ms | 738.8 FPS |

---

## 🔬 Key Features

- **Robust Anti-Overfitting Stack**: Integrated cosine learning rate decay, Gaussian blur and random erasing augmentation, gradient clipping, and label smoothing to fully mitigate the train-test performance gap.
- **Explainable AI (XAI)**: Utilizes Gradient-weighted Class Activation Mapping (Grad-CAM) to overlay heatmaps onto the structural margins of the tumor, visualising the model's region of focus for predictions.
- **Multiple Model Evaluation**: Out-of-the-box comparative testing for five architectures over identical schedules.

---

## 🛠 Project Structure

```
.
├── train.py                  # Core PyTorch training engine
├── run_all.sh                # Automation script for multi-model runs
├── gradcam.py                # Standalone Grad-CAM script for single image
├── gradcam_folder.py         # Folder-wise bulk Grad-CAM script for all classes
├── generate_summary_table.py # Compiles exact test accuracy and stats
├── results/                  # Accuracy curves, confusion matrices, and reports
│   └── gradcam/              # Visual heatmaps generated for all classes
└── data/                     # Training & testing MRI scans (ignored by git)
```

---

## 🚀 Getting Started

### 1. Prerequisites
Install all the core deep learning dependencies and the official Grad-CAM library:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn opencv-python grad-cam
```

### 2. Training Models
Run sequential, optimized 30-epoch training across multiple architectures:
```bash
bash run_all.sh
```

### 3. Generating Explainability Heatmaps
Generate folder-wise Grad-CAM heatmaps for any category and architecture:
```bash
python gradcam_folder.py
```

### 4. Compiling Results
To export your Markdown and LaTeX summary tables:
```bash
python generate_summary_table.py
```
