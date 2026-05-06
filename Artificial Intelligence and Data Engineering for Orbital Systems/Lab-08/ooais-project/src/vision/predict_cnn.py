from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.vision.cnn_model import SimpleCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


MODEL_PATH = Path("models/cnn_model.pt")
CLASS_NAMES_PATH = Path("models/cnn_classes.txt")
TEST_IMAGES_PATH = Path("data/processed/images/test/")


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        print(f"Error: class file not found: {CLASS_NAMES_PATH}")
        print("Train the model first.")
        raise SystemExit(1)
    
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]
        
    return class_names


def load_model(class_names):
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Train the model first.")
        raise SystemExit(1)
    
    model = SimpleCNN(num_classes=len(class_names))
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def predict_image(model, class_names, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: image not found: {path}")
        return
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    with Image.open(path) as image:
        image = image.convert("RGB")
        image_for_plot = image.copy()
        image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities,dim=1)
    
    predicted_class = class_names[
        predicted_index.item()
    ]
    
    print("=== CNN Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")
    
    out_dir = Path("results") / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prediction_{path.stem}.png"
    
    plt.imshow(image_for_plot)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence.item():.4f}")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization: {out_path}")


def prev_part():
    class_names = load_class_names()
    model = load_model(class_names)
    image_path = "data/processed/images/test/forest/forest_0000.jpg"
    predict_image(
        model,
        class_names,
        image_path
    )
    

def prepare_conf_mx():
    class_names = load_class_names()
    model = load_model(class_names)
    
    for img_class in ["forest", "residential", "river"]:
        img_class_path = TEST_IMAGES_PATH + img_class + "/"
    
    
    
    


def main():
    # prev_part()
    prepare_conf_mx()


if __name__ == "__main__":
    main()
