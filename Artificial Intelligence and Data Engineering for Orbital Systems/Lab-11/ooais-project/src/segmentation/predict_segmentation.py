from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.segmentation.unet_model import SmallUNet


MODEL_PATH = Path("models/small_unet.pt")
IMAGE_PATHS = [
    Path("data/segmentation/images/scene_0000.png"),
    Path("data/segmentation/images/scene_0010.png"),
    Path("data/segmentation/images/scene_0020.png"),
    Path("data/segmentation/images/scene_0030.png"),
    Path("data/segmentation/images/scene_0040.png"),
]
MASK_PATHS = [
    Path("data/segmentation/masks/scene_0000.png"),
    Path("data/segmentation/masks/scene_0010.png"),
    Path("data/segmentation/masks/scene_0020.png"),
    Path("data/segmentation/masks/scene_0030.png"),
    Path("data/segmentation/masks/scene_0040.png"),
]
OUTPUT_PATHS = [
    Path("reports/segmentation_examples/prediction_1.png"),
    Path("reports/segmentation_examples/prediction_2.png"),
    Path("reports/segmentation_examples/prediction_3.png"),
    Path("reports/segmentation_examples/prediction_4.png"),
    Path("reports/segmentation_examples/prediction_5.png"),
]
NUM_CLASSES = 4


PALETTE = {
0: (80, 80, 80),
1: (40, 140, 40),
2: (40, 80, 180),
3: (180, 180, 180)
}


def mask_to_rgb(mask):
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color
    
    return rgb


def load_model():
    if not MODEL_PATH.exists():
        print(f"Error: model not found: {MODEL_PATH}")
        print("Run train_segmentation.py first.")
        raise SystemExit(1)
    
    model = SmallUNet(num_classes=NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def predict_mask(model, image):
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1)
    return prediction[0].numpy()


def visualize(image, ground_truth, prediction, output_path):
    gt_rgb = mask_to_rgb(ground_truth)
    pred_rgb = mask_to_rgb(prediction)
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title("Input image")
    plt.axis("off")
    
    plt.subplot(1,3,2)
    plt.imshow(gt_rgb)
    plt.title("Ground truth mask")
    plt.axis("off")
    
    plt.subplot(1,3,3)
    plt.imshow(pred_rgb)
    plt.title("Predicted mask")
    plt.axis("off")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved prediction: {output_path}")


def main():
    model = load_model()
    for image_path, mask_path, output_path in zip(IMAGE_PATHS, MASK_PATHS, OUTPUT_PATHS):
        image = Image.open(image_path).convert("RGB")
        ground_truth = np.array(Image.open(mask_path), dtype=np.int64)
        prediction = predict_mask(model, image)
        visualize(image, ground_truth, prediction, output_path)


if __name__ == "__main__":
    main()

