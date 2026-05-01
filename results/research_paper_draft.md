# Comparative Analysis of Deep Learning Architectures for Robust Brain Tumor Classification and Explainable AI

**Author**: Vedant Rai
**Institutional Affiliation**: Advanced Machine Learning & Neuromedical Computing Group

---

## 1. Abstract
Automated classification of multi-class brain tumors from MRI scans remains a vital area of research due to the need for high-precision, low-latency, and explainable deep learning models. In this paper, we conduct a rigorous comparative study of five distinct deep convolutional neural network (CNN) architectures—ResNet50, InceptionV3, MobileNetV2, EfficientNet-B0, and a Custom 5-Block CNN—to analyze their real-world diagnostic performance under identical training regimes. To fully mitigate data-level overfitting, we implemented an optimized framework utilizing a **Cosine Annealing learning rate schedule, Label Smoothing regularization, and extensive data augmentation**. 

Our experimental outcomes show that **MobileNetV2** achieved the highest overall test accuracy of **95.31%** with a highly compact footprint (2.23M parameters, 8.74 MB), outperforming traditional massive networks. Furthermore, we integrated Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize internal feature extraction patterns. Our results demonstrate that lightweight networks exhibit highly localized, biologically sound features, confirming their massive potential for clinical diagnostic edge computers.

---

## 2. Introduction
Magnetic Resonance Imaging (MRI) is the gold standard for non-invasive diagnosis of cranial conditions. Accurately diagnosing brain tumors—such as **Gliomas, Meningiomas, and Pituitary tumors**—requires significant neuro-radiological expertise. Automated classification via convolutional neural networks (CNNs) has emerged as a promising method to accelerate diagnosis. 

However, translating these deep architectures into production clinical settings involves several key constraints:
1. **Clinical Edge Computing Constraints**: Hospital workstations and mobile devices are frequently compute-bound, necessitating small model footprints and minimal latency.
2. **The Overfitting Gap**: With limited availability of multi-class medical training scans, deep models are susceptible to memorizing training distributions. This results in poor accuracy on unseen testing sets.
3. **The Explainability Imperative**: Medical professionals require transparency. "Black-box" predictions limit clinical trust, making explainable artificial intelligence (XAI) critical.

To solve these issues, we fine-tuned and tested multiple models using identical parameters and applied Gradient-weighted Class Activation Mapping (Grad-CAM) to validate regional feature focus.

---

## 3. Methodology

```
                       [ Input MRI Scan ]
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        [ Training Data ]              [ Testing Data ]
                │                             │
    ┌───────────┴───────────┐                 │
    ▼                       ▼                 │
[ Augmentation ]    [ Regularization ]       │
- Random Flips      - Label Smoothing         │
- Gaussian Blur     - Gradient Clipping       │
- Random Erasing    - Cosine Decay            │
                │                             │
                ▼                             ▼
        [ Model Training ]            [ Model Evaluation ]
        - ResNet50                    - Test Accuracy (%)
        - InceptionV3                 - Parameters (M)
        - MobileNetV2                 - Latency (ms)
        - EfficientNet-B0             - Inference Speed (FPS)
        - Custom CNN                  - Grad-CAM Explainability
```

### 3.1 Data Augmentation & Preprocessing
Data diversity was augmented via random horizontal and vertical flips, rotations, color jittering, Gaussian Blur, and **Random Erasing** to simulate multiple imaging modalities and acquisition noise.

### 3.2 Advanced Regularizers
- **Learning Rate Decay**: A slow, smooth Cosine Annealing schedule ($5\times10^{-5} \to 1\times10^{-7}$) was utilized.
- **Label Smoothing**: Softened targets ($\epsilon=0.05$) to mitigate overconfidence.
- **Gradient Clipping**: Norm clamped at $1.0$ to ensure numerical stability.

---

## 4. Experimental Results

### 4.1 Quantitative Benchmarks
The final comparative metrics across all architectures are summarized in the following benchmark table:

| Model Architecture | Test Accuracy | Parameters | Model Size | Latency | Inference (FPS) |
|---|---|---|---|---|---|
| **MobileNetV2** | **95.31%** | **2.23M** | **8.74 MB** | **2.11 ms** | **474.1 FPS** |
| **EfficientNet-B0** | 95.12% | 4.01M | 15.59 MB | 3.20 ms | 312.3 FPS |
| **InceptionV3** | 94.19% | 25.12M | 96.16 MB | 6.07 ms | 164.7 FPS |
| **ResNet50** | 92.12% | 23.52M | 90.00 MB | 2.48 ms | 403.9 FPS |
| **Custom CNN** | 83.75% | 4.85M | 18.53 MB | 1.35 ms | 738.8 FPS |

MobileNetV2 demonstrated a brilliant accuracy of **95.31%**, which suggests that inverted residual blocks and depthwise separable convolutions are highly suited for tumor edge tracking in brain scans. 

---

## 5. Explainable AI via Grad-CAM Analysis

To bridge the interpretability gap, we generated folder-wise class activation maps for all the classes. This visualizes exactly which regions of the brain MRI the trained networks focused on to make their prediction.

### 5.1 Glioma Test Scan
Gliomas exhibit diffuse, poorly localized boundaries.

| ResNet50 | InceptionV3 | MobileNetV2 | EfficientNet-B0 |
|---|---|---|---|
| ![Glioma ResNet](gradcam/glioma_resnet.png) | ![Glioma Inception](gradcam/glioma_inception.png) | ![Glioma MobileNet](gradcam/glioma_mobilenet.png) | ![Glioma EfficientNet](gradcam/glioma_efficientnet.png) |

### 5.2 Meningioma Test Scan
Meningiomas are typically dural-based with distinct, well-defined borders.

| ResNet50 | InceptionV3 | MobileNetV2 | EfficientNet-B0 |
|---|---|---|---|
| ![Meningioma ResNet](gradcam/meningioma_resnet.png) | ![Meningioma Inception](gradcam/meningioma_inception.png) | ![Meningioma MobileNet](gradcam/meningioma_mobilenet.png) | ![Meningioma EfficientNet](gradcam/meningioma_efficientnet.png) |

---

## 6. Training Dynamics and Convergence Curves

To analyze the internal training stability, we track the accuracy and loss curves for ResNet50:

### A. Accuracy and Loss Tracking
![Accuracy Curve](resnet/accuracy.png)
![Loss Curve](resnet/loss.png)

### B. Confusion Matrix
![Confusion Matrix](resnet/confusion_matrix.png)

---

## 7. Conclusion
In this comprehensive comparative analysis, we evaluated five deep learning architectures on the multi-class brain tumor classification problem. We demonstrated that integrating specialized regularizers like Cosine scheduling, Gaussian data augmentation, and Label Smoothing prevents overfitting. Notably, our findings confirm that lightweight models like **MobileNetV2** perform with stellar diagnostic precision (**95.31%**) and process scans at incredible speeds (**474 FPS**), making them ideal for edge diagnostic applications.
