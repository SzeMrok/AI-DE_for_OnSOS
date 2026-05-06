from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
import numpy as np
import joblib

from src.vision.feature_extractor import extract_features

DATASET_DIR = Path("data/processed/images")

MODEL_PATH = Path("models/image_model.joblib")

def load_image_split(split_dir):
    X = []
    y = []
    
    class_dirs = sorted([
        path for path in split_dir.iterdir() 
        if path.is_dir()
    ])
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        image_files = sorted([
            path for path in class_dir.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        
        for image_path in image_files:
            with Image.open(image_path) as image:
                features = extract_features(image)
                X.append(features)
                y.append(class_name)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def load_training_and_test_data():
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"
    
    X_train, y_train = load_image_split(train_dir)
    X_test, y_test = load_image_split(test_dir)
    
    print("=== Image ML Dataset ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print("=== Training Image Classifier ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Number of features per image: {X_train.shape[1]}")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model trained.")

    return model

def evaluate_model(model, X_test, y_test):
    print("=== Model Evaluation ===")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Number of test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("=== Saving Model ===")
    print(f"Saved model: {MODEL_PATH}")


def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    results = []
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"=== Training {name} ===")
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        save_path = out_dir / f"{name}.joblib"
        joblib.dump(model, save_path)

        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} s")

        results.append({
            "model_name": name,
            "accuracy": accuracy,
            "training_time": training_time,
            "model": model,
            "path": str(save_path)
        })

    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / "accuracy_vs_training_time.png"

    times = [r["training_time"] for r in results]
    accs = [r["accuracy"] for r in results]
    labels = [r["model_name"] for r in results]

    plt.figure()
    plt.scatter(times, accs)
    for i, label in enumerate(labels):
        plt.annotate(label, (times[i], accs[i]))
    plt.xlabel("Training time (s)")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Training Time")
    plt.grid(True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot: {fig_path}")

    return results


def main():
    X_train, X_test, y_train, y_test = load_training_and_test_data()
    results = compare_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()


