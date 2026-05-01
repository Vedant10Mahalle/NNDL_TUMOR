# NNDL_TUMOR: Advanced Deep Learning & Explainable AI for Multi-Class Brain Tumor Classification

A highly optimized, production-ready, and explainable deep learning pipeline for the clinical diagnostic classification of magnetic resonance imaging (MRI) brain scans into four classes: **Glioma, Meningioma, Pituitary, and No Tumor**. This repository evaluates ResNet50, InceptionV3, MobileNetV2, EfficientNet-B0, and a Custom 5-Block CNN under matching conditions to analyze computational complexity versus classification efficacy.

---

## 🔬 1. Research Project Abstract
Automated, early detection of brain tumors from MRI scans is crucial for improving survival rates. In this comprehensive comparative analysis, we evaluate the real-world trade-offs between parameters, inference speed, disk footprint, and classification accuracy across five deep convolutional networks. Every model was trained over a fixed schedule of 30 epochs with identical multi-class configurations. To resolve data-level overfitting, we integrated an optimized training framework with **Cosine Annealing, Label Smoothing, and Gaussian Blur / Random Erasing transformations**. 

Our experimental outcomes demonstrate that **MobileNetV2** achieved the highest test accuracy of **95.31%** while possessing an incredibly compact footprint (2.23M parameters, 8.74 MB), outperforming heavier architectures like ResNet50 (92.12%) and InceptionV3 (94.19%). We leveraged Gradient-weighted Class Activation Mapping (**Grad-CAM**) to map internal convolutional features over structural MRI regions, validating that lightweight networks exhibit highly localized, biologically sound features.

---

## 📊 2. Master Performance Comparison

The final benchmarks illustrate the trade-off between model parameters, inference latency, and accuracy:

| Model Architecture | Test Accuracy | Total Parameters | Model Size | Latency | Inference Speed (FPS) |
|---|---|---|---|---|---|
| **MobileNetV2** | **95.31%** 🏆 | **2.23 Million** | **8.74 MB** | **2.11 ms** | **474.1 FPS** |
| **EfficientNet-B0** | 95.12% | 4.01 Million | 15.59 MB | 3.20 ms | 312.3 FPS |
| **InceptionV3** | 94.19% | 25.12 Million | 96.16 MB | 6.07 ms | 164.7 FPS |
| **ResNet50** | 92.12% | 23.52 Million | 90.00 MB | 2.48 ms | 403.9 FPS |
| **Custom CNN** | 83.75% | 4.85 Million | 18.53 MB | 1.35 ms | 738.8 FPS |

---

## 🛠 3. Advanced Anti-Overfitting Stack

Small datasets in clinical imaging frequently trigger overfitting, where the model achieves ~100% training accuracy but falls short on the independent test set. We implemented an end-to-end regularization stack to strictly control parameter tuning:

### A. Dynamic Learning Rate Scheduling & Optimization
Instead of a static learning rate, we utilize **Cosine Annealing Learning Rate Decay** starting at $5\times10^{-5}$ down to $1\times10^{-7}$. This prevents the model from aggressive early memorization and ensures it converges slowly and accurately.

### B. Label Smoothing Loss
By replacing traditional `CrossEntropyLoss()` with **Label Smoothing (0.05)**, we force the model to not predict absolute probabilities (e.g., 99.9% certainty). This directly limits the train-test accuracy gap.

### C. Enhanced Data Augmentation
To improve structural and environmental robustness, the pipeline subjects training images to random horizontal and vertical flips, rotations, color jittering, Gaussian Blur, and **Random Erasing (p=0.3)** to simulate variable MRI imaging scenarios.

---

## 🗺️ 4. Explainable AI: Folder-Wise Grad-CAM Visualizations

To validate that our models base their diagnostic predictions on correct visual features rather than background artifacts, we conducted Gradient-weighted Class Activation Mapping (Grad-CAM) analysis on testing images folder-by-folder.

### A. Glioma Tissue Activation
Glioma tumors generally exhibit diffuse borders. Grad-CAM shows how different models locate this specific type of lesion:

| ResNet50 | InceptionV3 | MobileNetV2 | EfficientNet-B0 |
|---|---|---|---|
| ![Glioma ResNet](results/gradcam/glioma_resnet.png) | ![Glioma Inception](results/gradcam/glioma_inception.png) | ![Glioma MobileNet](results/gradcam/glioma_mobilenet.png) | ![Glioma EfficientNet](results/gradcam/glioma_efficientnet.png) |

### B. Meningioma Tissue Activation
Meningiomas are typically dural-based and exhibit distinct, localized borders:

| ResNet50 | InceptionV3 | MobileNetV2 | EfficientNet-B0 |
|---|---|---|---|
| ![Meningioma ResNet](results/gradcam/meningioma_resnet.png) | ![Meningioma Inception](results/gradcam/meningioma_inception.png) | ![Meningioma MobileNet](results/gradcam/meningioma_mobilenet.png) | ![Meningioma EfficientNet](results/gradcam/meningioma_efficientnet.png) |

---

## 📈 5. Training Curves & Convergence

### ResNet50 Optimization Metrics

#### Accuracy and Loss Progression:
![ResNet Accuracy](results/resnet/accuracy.png)
![ResNet Loss](results/resnet/loss.png)

#### Confusion Matrix:
![ResNet Confusion Matrix](results/resnet/confusion_matrix.png)

---

## 💻 6. Project Architecture & Getting Started

```
.
├── train.py                  # Core PyTorch training script with optimization
├── run_all.sh                # Sequential training automation script
├── gradcam_folder.py         # Automates Grad-CAM visual heatmaps folder-wise
├── generate_summary_table.py # Compiles exact test accuracy and metrics
├── results/                  # Training curves, confusion matrices, and reports
│   └── gradcam/              # Explanatory visual heatmaps for each category
└── data/                     # Source MRI images (ignored by git)
```

### Reproduce the Results Locally
To install the dependencies, execute the full training loop, and generate tables:

```bash
# 1. Install prerequisites
pip install torch torchvision matplotlib seaborn scikit-learn opencv-python grad-cam

# 2. Run the multi-model optimization schedule
bash run_all.sh

# 3. Generate the summary comparisons and LaTeX table
python generate_summary_table.py

# 4. Create the Grad-CAM visualizations
python gradcam_folder.py
```
