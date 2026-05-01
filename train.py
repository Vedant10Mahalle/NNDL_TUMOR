import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # =========================
    # ARGUMENTS
    # =========================
    parser = argparse.ArgumentParser(description="Research-Grade CNN Training Script")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet", "inception", "mobilenet", "efficientnet", "custom"],
                        help="Model to use for training")
    args = parser.parse_args()

    # =========================
    # DEVICE
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    # PATHS & FOLDERS
    # =========================
    train_dir = "data/Training"
    test_dir  = "data/Testing"

    result_dir = f"results/{args.model}"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # =========================
    # HYPERPARAMETERS (Model-Specific)
    # =========================
    EPOCHS      = 30
    BATCH_SIZE  = 32
    NUM_WORKERS = 2

    if args.model in ["resnet", "inception", "efficientnet"]:
        DROPOUT      = 0.5
        WEIGHT_DECAY = 5e-4
        LR           = 5e-5      # Enough to converge, prevents overfitting
        LABEL_SMOOTH = 0.05      # Mild smoothing — helps generalization
    elif args.model == "mobilenet":
        DROPOUT      = 0.2
        WEIGHT_DECAY = 1e-4
        LR           = 1e-4      # Needs higher LR to converge in 30 epochs
        LABEL_SMOOTH = 0.05
    else:  # custom
        DROPOUT      = 0.5
        WEIGHT_DECAY = 1e-4
        LR           = 1e-4      # Custom CNN trains from scratch, needs higher LR
        LABEL_SMOOTH = 0.0       # No smoothing for from-scratch training

    # =========================
    # IMAGE SIZE
    # =========================
    img_size = 299 if args.model == "inception" else 224

    # =========================
    # TRANSFORMS (Stronger Augmentation)
    # =========================
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # =========================
    # DATA — Fair split for all models
    # =========================
    train_dataset_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset_full   = datasets.ImageFolder(train_dir, transform=val_transform)
    test_data          = datasets.ImageFolder(test_dir,  transform=val_transform)

    # Reproducible 70/30 train-val split
    torch.manual_seed(42)
    indices    = torch.randperm(len(train_dataset_full)).tolist()
    train_size = int(0.7 * len(train_dataset_full))

    train_data = torch.utils.data.Subset(train_dataset_full, indices[:train_size])
    val_data   = torch.utils.data.Subset(val_dataset_full,   indices[train_size:])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_names = train_dataset_full.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # =========================
    # MODEL SELECTION
    # =========================
    if args.model == "resnet":
        model = models.resnet50(weights="DEFAULT")
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif args.model == "inception":
        model = models.inception_v3(weights="DEFAULT", aux_logits=True)
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif args.model == "mobilenet":
        model = models.mobilenet_v2(weights="DEFAULT")
        model.classifier[1] = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(model.last_channel, num_classes)
        )

    elif args.model == "efficientnet":
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif args.model == "custom":
        # Deep 5-block CNN with Global Average Pooling — much more powerful than flat Flatten
        model = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),

            # Block 5: 256 → 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),

            # Global Average Pooling — reduces 512xHxW to 512 (no Flatten issues)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # Classifier head
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, num_classes)
        )

    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params:,} | Trainable: {trainable_params:,}")

    # =========================
    # LOSS + OPTIMIZER + SCHEDULER
    # =========================
    # Label smoothing — model-specific (prevents overconfidence)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine Annealing — the standard LR scheduler for CNN papers
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # =========================
    # TRAINING LOOP (Fixed 30 epochs)
    # =========================
    train_acc_list, val_acc_list   = [], []
    train_loss_list, val_loss_list = [], []
    epoch_times = []
    best_val_acc = 0.0

    print(f"\n--- Starting Training ({EPOCHS} epochs, lr={LR}) ---")
    start_total = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ===== TRAIN =====
        model.train()
        correct, total, train_loss_sum = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if args.model == "inception":
                outputs, aux_outputs = model(images)
                loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)

            loss.backward()
            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            _, pred  = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()

        train_acc  = 100 * correct / total
        train_loss = train_loss_sum / total
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # ===== VALIDATION =====
        model.eval()
        correct, total, val_loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                _, pred  = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (pred == labels).sum().item()

        val_acc  = 100 * correct / total
        val_loss = val_loss_sum / total
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Step LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Time: {epoch_time:.2f}s | LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/{args.model}.pth")
            print(f"  --> Best model saved (Val Acc: {best_val_acc:.2f}%)")

    total_train_time = time.time() - start_total

    # =========================
    # TEST EVALUATION
    # =========================
    print("\n--- Starting Testing ---")
    model.load_state_dict(torch.load(f"models/{args.model}.pth"))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # =========================
    # INFERENCE SPEED BENCHMARK
    # =========================
    print("\n--- Benchmarking Inference Speed ---")
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # GPU warm-up
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    latency_runs = 200
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(latency_runs):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    t1 = time.time()

    latency_ms = ((t1 - t0) / latency_runs) * 1000
    fps        = 1000 / latency_ms
    print(f"Inference Latency: {latency_ms:.2f} ms/image | Throughput: {fps:.1f} FPS")

    # =========================
    # MODEL SIZE
    # =========================
    model_size_mb = os.path.getsize(f"models/{args.model}.pth") / (1024 * 1024)

    # =========================
    # METRICS & REPORTS
    # =========================
    test_acc = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    report   = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm       = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    print("\nClassification Report:\n", report)

    # Save report.txt
    with open(f"{result_dir}/report.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nAccuracy per class:\n")
        for idx, cls_name in enumerate(class_names):
            f.write(f"{cls_name}: {per_class_acc[idx]:.4f}\n")

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.title(f"Confusion Matrix — {args.model.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_acc_list, label="Train Accuracy", linewidth=2)
    plt.plot(range(1, EPOCHS+1), val_acc_list,   label="Validation Accuracy", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title(f"Accuracy Curve — {args.model.upper()}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/accuracy.png", dpi=150)
    plt.close()

    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_loss_list, label="Train Loss", linewidth=2)
    plt.plot(range(1, EPOCHS+1), val_loss_list,   label="Validation Loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curve — {args.model.upper()}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/loss.png", dpi=150)
    plt.close()

    # Save metrics.txt
    with open(f"{result_dir}/metrics.txt", "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Final Train Accuracy:      {train_acc_list[-1]:.2f}%\n")
        f.write(f"Best Validation Accuracy:  {best_val_acc:.2f}%\n")
        f.write(f"Testing Accuracy:          {test_acc:.2f}%\n")
        f.write(f"Total Epochs:              {EPOCHS}\n")
        f.write(f"Average Epoch Time:        {np.mean(epoch_times):.2f} sec\n")
        f.write(f"Total Training Time:       {total_train_time / 60:.2f} minutes\n")
        f.write(f"\n--- Model Complexity ---\n")
        f.write(f"Total Parameters:          {total_params:,}\n")
        f.write(f"Trainable Parameters:      {trainable_params:,}\n")
        f.write(f"Model Size on Disk:        {model_size_mb:.2f} MB\n")
        f.write(f"\n--- Inference Speed ---\n")
        f.write(f"Latency:                   {latency_ms:.2f} ms/image\n")
        f.write(f"Throughput:                {fps:.1f} FPS\n")
        f.write(f"\n--- Hyperparameters ---\n")
        f.write(f"Learning Rate:             {LR}\n")
        f.write(f"Weight Decay:              {WEIGHT_DECAY}\n")
        f.write(f"Dropout Rate:              {DROPOUT}\n")
        f.write(f"Batch Size:                {BATCH_SIZE}\n")
        f.write(f"Optimizer:                 Adam\n")
        f.write(f"LR Scheduler:              CosineAnnealingLR\n")
        f.write(f"Loss Function:             CrossEntropyLoss (label_smoothing=0.1)\n")

    print("\n✅ FULL TRAINING + TEST + ANALYSIS DONE")
    print(f"Results saved in: {result_dir}/")
    print(f"Model saved:      models/{args.model}.pth")


if __name__ == "__main__":
    main()
