from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from src.vision.create_gradcam import *


def create_occlusion_map(
    model,
    image_tensor,
    predicted_class,
    patch_size=32,
    stride=16,
):
    _, _, H, W = image_tensor.shape

    sensitivity = np.zeros((H, W))

    with torch.no_grad():
        original_output = model(image_tensor)

    original_probability = (
        torch.softmax(original_output, dim=1)[0, predicted_class]
    ).item()

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = image_tensor.clone()

            occluded[
                :, :, y : y + patch_size, x : x + patch_size
            ] = 0

            with torch.no_grad():
                output = model(occluded)

            probability = (
                torch.softmax(output, dim=1)[0, predicted_class]
            ).item()

            drop = original_probability - probability

            sensitivity[
                y : y + patch_size, x : x + patch_size
            ] += drop

    sensitivity -= sensitivity.min()
    sensitivity /= (sensitivity.max() + 1e-8)

    return sensitivity


def visualize_occlusion(
    image,
    sensitivity,
):
    image = image.resize((224, 224))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sensitivity, cmap="jet")
    plt.title("Occlusion Sensitivity")
    plt.axis("off")

    plt.tight_layout()

    out_path = Path("reports/gradcam_examples/occlusion.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)


def main():
    class_names = load_class_names()

    model = load_model(class_names)

    image_path = Path("data/processed/images/test/river/river_0000.jpg")

    image, image_tensor = load_image(image_path)

    predicted_class = predict(
        model,
        image_tensor,
        class_names,
    )

    sensitivity = create_occlusion_map(
        model,
        image_tensor,
        predicted_class,
    )

    visualize_occlusion(
        image,
        sensitivity,
    )


if __name__ == "__main__":
    main()
