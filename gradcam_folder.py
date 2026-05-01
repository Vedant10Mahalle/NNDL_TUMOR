import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Official GradCAM library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_model(model_name, num_classes, device):
    img_size = 299 if model_name == "inception" else 224

    if model_name == "resnet":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
        target_layers = [model.layer4[-1]]

    elif model_name == "inception":
        model = models.inception_v3(weights=None, aux_logits=True)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
        target_layers = [model.Mixed_7c.branch_pool]

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, num_classes))
        target_layers = [model.features[-1]]

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.classifier[1].in_features, num_classes))
        target_layers = [model.features[-1]]

    model_path = f"models/{model_name}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing weights for {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, target_layers, img_size


def generate_gradcam_visual(model, target_layers, img_size, img_path, out_path, device):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    np_img = np.array(pil_img.resize((img_size, img_size))).astype(np.float32) / 255.0

    cam = GradCAM(model=model, target_layers=target_layers)

    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        pred_idx = torch.argmax(output, dim=1).item()

    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np_img)
    plt.title("Original MRI Scan")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM ({class_names[pred_idx].capitalize()})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dir = "data/Testing"
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    classes = sorted(os.listdir(test_dir))
    models_list = ["resnet", "inception", "mobilenet", "efficientnet"]

    os.makedirs("results/gradcam", exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        # Get first available image from the directory
        images = sorted(os.listdir(cls_dir))
        if not images:
            print(f"No images found for class folder: {cls}")
            continue

        first_img = os.path.join(cls_dir, images[0])
        print(f"\nProcessing class folder '{cls}' using first available image: {first_img}")

        for model_name in models_list:
            print(f"  Generating Grad-CAM for {model_name}...")
            try:
                model, target_layers, img_size = load_model(model_name, len(classes), device)
                out_path = f"results/gradcam/{cls}_{model_name}.png"
                generate_gradcam_visual(model, target_layers, img_size, first_img, out_path, device)
                print(f"  ✅ Saved: {out_path}")
            except Exception as e:
                print(f"  ❌ Error for {model_name} on {cls}: {e}")

    print("\n🎉 ALL FOLDER-WISE GRAD-CAM VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("Files saved to results/gradcam/ directory.")


if __name__ == "__main__":
    main()
