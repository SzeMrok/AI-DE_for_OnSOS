from pathlib import Path
from PIL import Image
import joblib
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.vision.feature_extractor import extract_features

MODEL_PATH = Path("models/image_model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        raise SystemExit(1)

    model = joblib.load(MODEL_PATH)

    print("Model loaded.")

    return model


def predict_image(model, image_path):
    path = Path(image_path)

    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return

    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.copy()

    prediction = model.predict([features])[0]

    print("=== Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {prediction}")

    # Prepare output directory and file
    out_dir = Path("results") / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prediction_{path.stem}.png"

    plt.imshow(image_for_plot)
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization: {out_path}")


def main():
    model = load_model()

    parser = argparse.ArgumentParser(description="Predict single image class and save visualization")
    parser.add_argument('image', nargs='?', default="data/processed/images/test/forest/forest_0000.jpg",
                        help='Path to image to predict')
    args = parser.parse_args()

    predict_image(model, args.image)


if __name__ == "__main__":
    main()
