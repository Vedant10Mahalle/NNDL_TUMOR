# Comparative Analysis of Deep Learning Architectures for Robust Brain Tumor Classification and Explainable AI

## 1. Abstract
Accurate and automated classification of brain tumors from Magnetic Resonance Imaging (MRI) is essential for timely medical intervention. In this paper, we conduct a rigorous comparative study of five distinct deep convolutional neural network (CNN) architectures—ResNet50, InceptionV3, MobileNetV2, EfficientNet-B0, and a deeper custom 5-block CNN—to evaluate their efficacy for clinical applications. Each model was evaluated under identical conditions over a fixed 30-epoch training schedule utilizing data augmentation, cosine learning rate decay, and label smoothing to fully mitigate overfitting. Our experimental results show that MobileNetV2 achieved the highest test accuracy of **95.31%** while possessing a highly compact footprint (2.23M parameters, 8.74 MB on disk). Furthermore, we utilized Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize and compare regional activations. This comprehensive study demonstrates that lightweight, highly efficient networks can outperform heavy architectures, suggesting an optimal approach for real-time edge deployment in hospital diagnostic workflows.

---

## 2. Introduction
Brain tumors, including gliomas, meningiomas, and pituitary tumors, pose serious health threats worldwide. Diagnosing these conditions typically requires skilled radiologists to interpret complex MRI scans. In recent years, deep learning and convolutional neural networks (CNNs) have achieved outstanding diagnostic performance. 

Despite their success, translating these models into actual clinical workflows introduces significant challenges:
1. **Computational Constraints**: Medical systems often operate on edge devices with limited memory and inference power. 
2. **Generalization & Overfitting**: Small datasets cause overconfidence in training accuracy, resulting in poor validation performance.
3. **Black-box Problem**: Deep networks lack explainability, making it hard for clinicians to trust their outputs.

To address these limitations, we evaluate multiple CNN architectures under exactly matching training paradigms and provide visual class activation maps to highlight exactly where the networks base their diagnostic predictions.

---

## 3. Methodology

### 3.1 Dataset Overview
The dataset consists of multi-class MRI images categorized into 4 classes: `glioma`, `meningioma`, `notumor`, and `pituitary`. The dataset is split into 70% for training/validation and 30% for independent testing.

### 3.2 Model Architectures Evaluated
1. **ResNet50**: Incorporates skip connections to train deep networks.
2. **InceptionV3**: Features multi-scale kernel convolutions for extracting features of varying spatial sizes.
3. **MobileNetV2**: Employs inverted residual blocks and depthwise separable convolutions to dramatically minimize parameters.
4. **EfficientNet-B0**: Optimizes network depth, width, and resolution scaling concurrently.
5. **Custom CNN**: A hand-crafted, deeper 5-block CNN featuring 10 convolutional layers, batch normalization, dropout, and global average pooling to prevent overfitting when learning features from scratch.

### 3.3 Training Strategy and Regularization
All models were fine-tuned via identical settings to ensure a 100% fair scientific comparison:
- **Learning Rate Schedule**: Cosine Annealing learning rate starting at $5\times10^{-5}$ down to $1\times10^{-7}$ to avoid aggressive early memorization.
- **Data Augmentation**: Random horizontal and vertical flips, rotations ($20^\circ$), Gaussian blur, color jittering, and random erasing ($p=0.3$) to expand structural variance.
- **Label Smoothing**: Applied smoothing factor ($\epsilon = 0.05$) to prevent the network from predicting overconfident probabilities.
- **Gradient Clipping**: Norm clamped at $1.0$ to ensure stable weight updates.

---

## 4. Experimental Results

### 4.1 Quantitative Performance

The following table summarizes the test accuracy, computational complexity, and inference latency for all evaluated models:

| Model Architecture | Test Accuracy | Total Parameters | Model Size | Latency | Inference Speed |
|---|---|---|---|---|---|
| **MobileNetV2** | **95.31%** | **2.23 Million** | **8.74 MB** | **2.11 ms** | **474.1 FPS** |
| **EfficientNet-B0** | 95.12% | 4.01 Million | 15.59 MB | 3.20 ms | 312.3 FPS |
| **InceptionV3** | 94.19% | 25.12 Million | 96.16 MB | 6.07 ms | 164.7 FPS |
| **ResNet50** | 92.12% | 23.52 Million | 90.00 MB | 2.48 ms | 403.9 FPS |
| **Custom CNN** | 83.75% | 4.85 Million | 18.53 MB | 1.35 ms | 738.8 FPS |

Our benchmarks reveal that **MobileNetV2** achieved the best overall accuracy at **95.31%**, exceeding both ResNet50 and InceptionV3. This highlights the effectiveness of depthwise separable convolutions on this specific classification task. Furthermore, the Custom CNN achieved a solid **83.75%** from scratch, highlighting its strong baseline capability.

---

## 5. Explainable AI via Grad-CAM Analysis
To demystify the black-box nature of our deep CNNs, we integrated **Gradient-weighted Class Activation Mapping (Grad-CAM)**. Grad-CAM visualizes the gradients flowing into the final convolutional layer of the network to produce a coarse localization map highlighting the important regions of the brain MRI used for prediction.

```
results/gradcam/
├── glioma_resnet.png           <-- Focuses on broad global activations
├── glioma_inception.png        <-- Accurately binds internal tumor edges
├── glioma_mobilenet.png        <-- Captures granular high-contrast tissue
└── glioma_efficientnet.png     <-- Highly localized on primary lesions
```

Our qualitative analysis demonstrates that:
1. **ResNet50 and InceptionV3** tend to consider large areas around the tumor, providing wide contextual heatmaps.
2. **MobileNetV2 and EfficientNet-B0** show distinct, sharp localization precisely over the structural margins of the tumor itself. This specific focus explains their higher testing accuracy (95.31% and 95.12%, respectively) as they are less influenced by background tissue.

---

## 6. Conclusion
In this research, we comprehensively evaluated five deep learning architectures on the multi-class brain tumor classification problem. We demonstrated that with proper regularizers, including cosine scheduling, data augmentation, and mild label smoothing, model overfitting is mitigated effectively. Most notably, **MobileNetV2** not only requires much less storage space (8.74 MB) and compute capacity, but it also outperforms massive networks with a test accuracy of **95.31%**. Combined with explainable Grad-CAM visualizations, our results confirm that lightweight CNN models provide highly efficient, real-world deployment value for radiologists and diagnostic edge computers.
