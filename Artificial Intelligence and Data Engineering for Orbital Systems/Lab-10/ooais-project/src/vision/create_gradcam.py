from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM as TorchGradCAM, HiResCAM, EigenCAM, LayerCAM

MODEL_PATH = Path("models/resnet18_transfer.pt")
CLASS_NAMES_PATH = Path("models/resnet18_classes.txt")

IMAGE_PATH = Path("data/processed/images/test/river/river_0000.jpg")
IMAGE_PATH_FOREST = Path("data/processed/images/test/forest/forest_0000.jpg")
IMAGE_PATH_RESIDENTIAL = Path("data/processed/images/test/residential/residential_0000.jpg")
IMAGE_PATH_NOISE = Path("data/inference_samples/noise.jpg")
IMAGE_PATH_HIGHWAY = Path("data/raw/eurosat/2750/Highway/Highway_1.jpg")
OUTPUT_PATH = Path("reports/gradcam_example.png")


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        print(f"Error: class file not found: {CLASS_NAMES_PATH}")
        print("Train the transfer model first.")
        raise SystemExit(1)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [
            line.strip()
            for line in f
        ]

    return class_names


def load_model(class_names):
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Train the transfer model first.")
        raise SystemExit(1)

    model = models.resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(class_names))

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        raise SystemExit(1)
    
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    tensor = transform(image)

    return image, tensor.unsqueeze(0)


class ManualGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activations
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradients
        )

    def save_activations(self, module, input_data, output_data):
        self.activations = output_data.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class_index):
        self.model.zero_grad()

        outputs = self.model(image_tensor)
        score = outputs[0, target_class_index]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for channel_index, weight in enumerate(weights):
            cam += weight * activations[channel_index]

        cam = torch.relu(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.numpy()

    def close(self):
        try:
            self.forward_hook.remove()
        except Exception:
            pass
        try:
            self.backward_hook.remove()
        except Exception:
            pass


def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted.item()]
    
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")
    
    return predicted.item()

def create_heatmap(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class_index = torch.argmax(outputs, dim=1).item()

    cam = ManualGradCAM(model=model, target_layer=model.layer4[-1])

    try:
        grayscale_cam = cam.generate(image_tensor, predicted_class_index)
    finally:
        cam.close()

    return grayscale_cam

def visualize(image, heatmap):
    output_path = Path(f"reports/gradcam_examples/example_{IMAGE_PATH_HIGHWAY.stem}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = image.resize((224,224))
    image_array = (np.array(image).astype(np.float32)/255)
    heatmap_image = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
    heatmap_array = np.array(heatmap_image.resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255.0
    visualization = show_cam_on_image(
        image_array,
        heatmap_array,
        use_rgb=True
    )
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(visualization)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    

def create_cam_by_method(model, image_tensor, method_name):
    target_layers = [model.layer4[-1]]
    methods = {
        "GradCAM": TorchGradCAM,
        "HiResCAM": HiResCAM,
        "EigenCAM": EigenCAM,
        "LayerCAM": LayerCAM
    }
    cam_class = methods[method_name]
    cam = cam_class(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam[0]



def visualize_multiple_cams(image, heatmaps):
    image = image.resize((224,224))
    image_array = (np.array(image).astype(np.float32)/255.0)
    plt.figure(figsize=(16,8))
    plt.subplot(2,3,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")
    index = 2
    for method_name, heatmap in heatmaps.items():
        visualization = show_cam_on_image(image_array, heatmap, use_rgb=True)
        plt.subplot(2, 3, index)
        plt.imshow(visualization)
        plt.title(method_name)
        plt.axis("off")
        index += 1
    plt.tight_layout()
    output_path = ("reports/gradcam_examples/cam_methods_comparison.png")
    plt.savefig(output_path)
    print(f"Saved: {output_path}")


def main():
    class_names = (load_class_names())
    model = load_model(class_names)
    image, tensor = (load_image(IMAGE_PATH_HIGHWAY))
    predict(
        model,
        tensor,
        class_names
    )
    heatmap = (create_heatmap(model, tensor))
    visualize(image, heatmap)


if __name__ == "__main__":
    main()
