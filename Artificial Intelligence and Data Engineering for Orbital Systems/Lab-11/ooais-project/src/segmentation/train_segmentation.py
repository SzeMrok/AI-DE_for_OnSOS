from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from src.segmentation.segmentation_dataset import SyntheticSegmentationDataset
from src.segmentation.unet_model import SmallUNet


IMAGE_DIR = Path("data/segmentation/images")
MASK_DIR = Path("data/segmentation/masks")
MODEL_PATH = Path("models/small_unet.pt")
REPORT_PATH = Path("reports/segmentation_report.txt")
CLASS_WISE_REPORT_PATH = Path("reports/class_wise_accuracy.txt")
CLASS_NAMES = ["background", "vegetation", "water", "urban"]

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4
RANDOM_SEED = 42


def create_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SyntheticSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transform=transform
    )
    train_size = int(0.8 * len(dataset))
    test_size = (len(dataset)- train_size)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    print("=== Segmentation DataLoaders ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    images, masks = next(
    iter(train_loader)
    )
    print(f"Batch image shape: {images.shape}")
    print(f"Batch mask shape: {masks.shape}")
    
    return train_loader, test_loader


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, train_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = (total_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {average_loss:.4f}")


def evaluate_model(model, test_loader, device):
    model.eval()
    correct_pixels = 0
    total_pixels = 0
    class_correct = torch.zeros(NUM_CLASSES)
    class_total = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
            for class_id in range(NUM_CLASSES):
                class_mask = masks == class_id
                class_correct[class_id] += (predictions[class_mask] == class_id).sum().item()
                class_total[class_id] += class_mask.sum().item()
    
    accuracy = (correct_pixels / total_pixels)
    class_accuracies = {}
    for class_id in range(NUM_CLASSES):
        if class_total[class_id] > 0:
            class_accuracies[class_id] = class_correct[class_id] / class_total[class_id]
        else:
            class_accuracies[class_id] = 0.0
    
    print("=== Segmentation Evaluation ===")
    print(f"Pixel accuracy: {accuracy:.4f}")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name} accuracy: {class_accuracies[class_id]:.4f}")
    
    return accuracy, class_accuracies


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")


def save_class_wise_report(accuracy, class_accuracies):
    best_class_id = max(class_accuracies, key=class_accuracies.get)
    worst_class_id = min(class_accuracies, key=class_accuracies.get)
    CLASS_WISE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_WISE_REPORT_PATH, "w") as f:
        f.write("CLASS-WISE PIXEL ACCURACY\n")
        f.write("=========================\n\n")
        f.write(f"Global pixel accuracy: {accuracy:.4f}\n\n")
        for class_id, class_name in enumerate(CLASS_NAMES):
            f.write(f"{class_name}: {class_accuracies[class_id]:.4f}\n")
        f.write("\nWhich class achieved the best accuracy?\n")
        f.write(f"{CLASS_NAMES[best_class_id]} ({class_accuracies[best_class_id]:.4f})\n\n")
        f.write("Which class achieved the worst accuracy?\n")
        f.write(f"{CLASS_NAMES[worst_class_id]} ({class_accuracies[worst_class_id]:.4f})\n\n")
        f.write("Is global pixel accuracy misleading?\n")
        f.write("\n\n")
        f.write("Why are small classes often harder to evaluate?\n")
        f.write("\n\n")
    print(f"Saved report: {CLASS_WISE_REPORT_PATH}")


def save_report(accuracy):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(REPORT_PATH, "w") as f:
        f.write("SEMANTIC SEGMENTATION REPORT\n")
        f.write("============================\n\n")
        f.write("Model: Small U-Net\n")
        f.write("Dataset: synthetic EO segmentation dataset\n")
        f.write(f"Classes: {NUM_CLASSES}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Pixel accuracy: {accuracy:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write("The model was trained to assign a land-cover class to every pixel in the image.\n")
    print(f"Saved report: {REPORT_PATH}")


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    train_loader, test_loader = (create_dataloaders())
    device = get_device()
    print(f"Using device: {device}")
    model = SmallUNet(num_classes=NUM_CLASSES)
    model = model.to(device)
    train_model(model, train_loader, device)
    accuracy, class_accuracies = evaluate_model(model, test_loader, device)
    
    save_model(model)
    save_report(accuracy)
    save_class_wise_report(accuracy, class_accuracies)


if __name__ == "__main__":
    main()
