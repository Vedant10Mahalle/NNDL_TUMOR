import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os

# Official GradCAM library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Visualizations")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["resnet", "inception", "mobilenet", "efficientnet"],
                        help="Model to use")
    parser.add_argument("--image", type=str, required=True, help="Path to input MRI image")
    parser.add_argument("--out", type=str, default="gradcam_output.png", help="Path to save visual")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 1. LOAD MODEL & SET TARGET LAYER
    # ==========================================
    num_classes = 4
    img_size = 299 if args.model == "inception" else 224

    if args.model == "resnet":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
        target_layers = [model.layer4[-1]]  # Last Conv Layer of ResNet50

    elif args.model == "inception":
        model = models.inception_v3(weights=None, aux_logits=True)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
        target_layers = [model.Mixed_7c.branch_pool]  # A top mixed convolutional layer

    elif args.model == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, num_classes))
        target_layers = [model.features[-1]]  # Last conv/features layer

    elif args.model == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.classifier[1].in_features, num_classes))
        target_layers = [model.features[-1]]  # Last conv/features layer

    # Load weights
    model_path = f"models/{args.model}.pth"
    if not os.path.exists(model_path):
        print(f"Error: Trained model file {model_path} not found. Please train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ==========================================
    # 2. PREPROCESS IMAGE
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    pil_img = Image.open(args.image).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Convert PIL to standard numpy for CV2 plotting later
    np_img = np.array(pil_img.resize((img_size, img_size))).astype(np.float32) / 255.0

    # ==========================================
    # 3. COMPUTE GRAD-CAM
    # ==========================================
    cam = GradCAM(model=model, target_layers=target_layers)

    # Generate prediction to choose target class
    with torch.no_grad():
        output = model(input_tensor)
        # Handle inception's train/eval behavior if needed
        if isinstance(output, tuple):
            output = output[0]
        pred_idx = torch.argmax(output, dim=1).item()

    # Target class label
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    print(f"Predicted Class: {class_names[pred_idx]} (Class Index: {pred_idx})")

    # Target activation for the predicted class
    targets = [ClassifierOutputTarget(pred_idx)]
    
    # Compute heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Overlay heatmap onto the RGB original image
    visualization = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)

    # ==========================================
    # 4. SAVE & DISPLAY RESULTS
    # ==========================================
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np_img)
    plt.title("Original MRI Scan")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM Heatmap\n(Focusing on {class_names[pred_idx]})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()

    print(f"✅ Success! Grad-CAM output saved in: {args.out}")

if __name__ == "__main__":
    main()
