from pathlib import Path
import torch
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.vision.cnn_model import SimpleCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from src.vision.image_dataset import EuroSATDataset


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
    

# task 11
def prepare_conf_mx():
    class_names = load_class_names()
    model = load_model(class_names)

    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    y_true = []
    y_pred = []

    for class_name in class_names:
        class_dir = TEST_IMAGES_PATH / class_name
        if not class_dir.exists():
            continue

        for image_path in sorted(class_dir.glob("*")):
            if not image_path.is_file():
                continue

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image_tensor)
                predicted_index = torch.argmax(outputs, dim=1).item()

            y_true.append(class_to_index[class_name])
            y_pred.append(predicted_index)

    if not y_true:
        print(f"No test images found in: {TEST_IMAGES_PATH}")
        return

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(
        ax=ax,
        cmap="viridis",
        values_format="d",
        colorbar=True
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confusion_matrix.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix: {out_path}")
    

# task 12
def compare_cnn_vs_ml():
    TRAIN_DIR = Path("data/processed/images/train")
    TEST_DIR = TEST_IMAGES_PATH
    class_names = load_class_names()
    model = load_model(class_names)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    def load_split(dir_path):
        X = []
        y = []
        file_paths = []
        for cls_idx, cls in enumerate(class_names):
            cls_dir = Path(dir_path) / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.glob("*")):
                if not p.is_file():
                    continue
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    t = transform(im)
                    X.append(t.numpy().transpose(1, 2, 0).ravel())
                    y.append(cls_idx)
                    file_paths.append(p)
        return np.stack(X) if X else np.zeros((0,)), np.array(y), file_paths

    X_train, y_train, _ = load_split(TRAIN_DIR)
    X_test, y_test, test_paths = load_split(TEST_DIR)

    if X_train.size == 0 or X_test.size == 0:
        print("No train/test images found for classical ML comparison.")
        return

    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    y_pred_rf = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)

    y_pred_cnn = []
    for p in test_paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            tensor = transform(im).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
            pred_idx = int(torch.argmax(out, dim=1).item())
        y_pred_cnn.append(pred_idx)
    y_pred_cnn = np.array(y_pred_cnn)
    cnn_acc = accuracy_score(y_test, y_pred_cnn)

    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=list(range(len(class_names))))
    cm_cnn = confusion_matrix(y_test, y_pred_cnn, labels=list(range(len(class_names))))

    rf_confusions = cm_rf.sum() - np.trace(cm_rf)
    cnn_confusions = cm_cnn.sum() - np.trace(cm_cnn)

    rf_per_class = np.diag(cm_rf) / cm_rf.sum(axis=1)
    cnn_per_class = np.diag(cm_cnn) / cm_cnn.sum(axis=1)

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=class_names)
    disp.plot(ax=ax, cmap="viridis", values_format="d", colorbar=False)
    ax.set_title("Random Forest Confusion Matrix")
    plt.tight_layout()
    rf_cm_path = out_dir / "confusion_rf.png"
    plt.savefig(rf_cm_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=class_names)
    disp.plot(ax=ax, cmap="viridis", values_format="d", colorbar=False)
    ax.set_title("CNN Confusion Matrix")
    plt.tight_layout()
    cnn_cm_path = out_dir / "confusion_cnn.png"
    plt.savefig(cnn_cm_path, bbox_inches="tight")
    plt.close(fig)

    def save_example_images(y_true, y_pred, paths, prefix):
        correct_saved = False
        incorrect_saved = False
        for true, pred, p in zip(y_true, y_pred, paths):
            if true == pred and not correct_saved:
                with Image.open(p) as im:
                    im.save(out_dir / f"{prefix}_correct_{p.stem}.png")
                correct_saved = True
            if true != pred and not incorrect_saved:
                with Image.open(p) as im:
                    im.save(out_dir / f"{prefix}_incorrect_{p.stem}.png")
                incorrect_saved = True
            if correct_saved and incorrect_saved:
                break

    save_example_images(y_test, y_pred_rf, test_paths, "rf")
    save_example_images(y_test, y_pred_cnn, test_paths, "cnn")

    analysis_path = out_dir / "cnn_vs_ml.txt"
    with open(analysis_path, "w") as f:
        f.write("CNN VS CLASSICAL ML\n")
        f.write("===================\n\n")
        
        f.write("CLASSICAL ML:\n")
        f.write("Model: Random Forest\n")
        f.write(f"Training time: {rf_train_time:.3f}s\n")
        f.write(f"Accuracy: {rf_acc:.4f}\n\n")
        
        f.write("CNN:\n")
        f.write("Model: Simple CNN\n")
        f.write("Training time: ?s\n")
        f.write(f"Accuracy: {cnn_acc:.4f}\n\n")
        
        f.write("BETTER ACCURACY:\n")
        if cnn_acc > rf_acc:
            f.write(f"CNN ({cnn_acc:.4f} vs {rf_acc:.4f}, +{cnn_acc-rf_acc:.4f})\n\n")
        elif cnn_acc < rf_acc:
            f.write(f"Random Forest ({rf_acc:.4f} vs {cnn_acc:.4f}, +{rf_acc-cnn_acc:.4f})\n\n")
        else:
            f.write("Tie\n\n")
        
        f.write("FASTER TRAINING:\n")
        f.write(f"Random Forest trained in {rf_train_time:.3f}s\n")
        f.write("CNN training time from original training script ?s\n")
        
        f.write("\nFEWER CLASS CONFUSIONS:\n")
        if rf_confusions < cnn_confusions:
            f.write(f"Random Forest ({rf_confusions} misclassifications vs {cnn_confusions})\n\n")
        elif cnn_confusions < rf_confusions:
            f.write(f"CNN ({cnn_confusions} misclassifications vs {rf_confusions})\n\n")
        else:
            f.write("Tie\n\n")
        
        f.write("PER-CLASS ACCURACY:\n")
        f.write("Random Forest:\n")
        for i, cls in enumerate(class_names):
            f.write(f"  {cls}: {rf_per_class[i]:.4f}\n")
        f.write("\nCNN:\n")
        for i, cls in enumerate(class_names):
            f.write(f"  {cls}: {cnn_per_class[i]:.4f}\n")
        f.write("\n")
        
        f.write("GENERALIZATION:\n")
        rf_variance = np.var(rf_per_class)
        cnn_variance = np.var(cnn_per_class)
        f.write(f"Random Forest per-class accuracy variance: {rf_variance:.6f}\n")
        f.write(f"CNN per-class accuracy variance: {cnn_variance:.6f}\n")
        if cnn_variance < rf_variance:
            f.write("CNN shows more consistent performance (lower variance).\n")
        elif rf_variance < cnn_variance:
            f.write("Random Forest shows more consistent performance (lower variance).\n")
        f.write(f"\nHardest class for RF: {class_names[np.argmin(rf_per_class)]} ({np.min(rf_per_class):.4f})\n")
        f.write(f"Hardest class for CNN: {class_names[np.argmin(cnn_per_class)]} ({np.min(cnn_per_class):.4f})\n")
        
        f.write(f"\nConfusion matrices saved:\n")
        f.write(f"- RF: {rf_cm_path}\n")
        f.write(f"- CNN: {cnn_cm_path}\n")
        f.write(f"\nExample images saved in {out_dir}/\n")

    print(f"Saved comprehensive comparison report: {analysis_path}")
    print(f"RF: {rf_acc:.4f} acc, {rf_train_time:.3f}s | CNN: {cnn_acc:.4f} acc")
    

# task 13
def train_with_augmentation():    
    TRAIN_DIR = Path("data/processed/images/train")
    TEST_DIR = TEST_IMAGES_PATH
    BATCH_SIZE = 16
    EPOCHS = 8
    LEARNING_RATE = 0.001
    
    class_names = load_class_names()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_baseline = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    transform_augmented = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    
    
    test_dataset = EuroSATDataset(
        root_dir=TEST_DIR,
        transform=transform_baseline
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    def train_and_eval(model, train_transform, model_name):
        train_dataset = EuroSATDataset(
            root_dir=TRAIN_DIR,
            transform=train_transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        
        model = model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        epoch_losses = []
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f"{model_name} - Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.numpy())
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        
        return epoch_losses, accuracy, cm, per_class_acc, y_pred, y_true
    
    print("\n=== Training Baseline (No Augmentation) ===")
    model_baseline = SimpleCNN(num_classes=len(class_names))
    baseline_losses, baseline_acc, cm_baseline, baseline_per_class, _, _ = train_and_eval(
        model_baseline, transform_baseline, "Baseline"
    )
    
    print("\n=== Training With Augmentation ===")
    model_augmented = SimpleCNN(num_classes=len(class_names))
    aug_losses, aug_acc, cm_aug, aug_per_class, _, _ = train_and_eval(
        model_augmented, transform_augmented, "Augmented"
    )
    
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=class_names)
    disp.plot(ax=ax, cmap="viridis", values_format="d", colorbar=False)
    ax.set_title("Baseline (No Augmentation)")
    plt.tight_layout()
    baseline_cm_path = out_dir / "confusion_baseline.png"
    plt.savefig(baseline_cm_path, bbox_inches="tight")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_aug, display_labels=class_names)
    disp.plot(ax=ax, cmap="viridis", values_format="d", colorbar=False)
    ax.set_title("With Augmentation")
    plt.tight_layout()
    aug_cm_path = out_dir / "confusion_augmented.png"
    plt.savefig(aug_cm_path, bbox_inches="tight")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(baseline_losses, label="Baseline (No Augmentation)", marker='o')
    ax.plot(aug_losses, label="With Augmentation", marker='s')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = out_dir / "training_loss_comparison.png"
    plt.savefig(loss_plot_path, bbox_inches="tight")
    plt.close(fig)
    
    baseline_loss_variance = np.var(baseline_losses)
    aug_loss_variance = np.var(aug_losses)
    
    per_class_improvements = aug_per_class - baseline_per_class
    best_improved_class = class_names[np.argmax(per_class_improvements)]
    worst_affected_class = class_names[np.argmin(per_class_improvements)]
    
    analysis_path = out_dir / "data_augmentation.txt"
    with open(analysis_path, "w") as f:
        f.write("DATA AUGMENTATION ANALYSIS\n")
        f.write("==========================\n\n")
        
        f.write("BASELINE (No Augmentation):\n")
        f.write(f"Final Accuracy: {baseline_acc:.4f}\n")
        f.write(f"Training Loss Variance: {baseline_loss_variance:.6f}\n")
        f.write(f"Final Epoch Loss: {baseline_losses[-1]:.4f}\n\n")
        
        f.write("WITH AUGMENTATION:\n")
        f.write(f"Final Accuracy: {aug_acc:.4f}\n")
        f.write(f"Training Loss Variance: {aug_loss_variance:.6f}\n")
        f.write(f"Final Epoch Loss: {aug_losses[-1]:.4f}\n\n")
        
        f.write("DID AUGMENTATION IMPROVE ACCURACY?\n")
        acc_diff = aug_acc - baseline_acc
        if acc_diff > 0:
            f.write(f"YES: +{acc_diff:.4f} ({baseline_acc:.4f} -> {aug_acc:.4f})\n\n")
        elif acc_diff < 0:
            f.write(f"NO: {acc_diff:.4f} ({baseline_acc:.4f} -> {aug_acc:.4f})\n\n")
        else:
            f.write(f"NO CHANGE: {baseline_acc:.4f}\n\n")
        
        f.write("DID TRAINING BECOME MORE STABLE?\n")
        if aug_loss_variance < baseline_loss_variance:
            f.write(f"YES: {baseline_loss_variance:.6f} to {aug_loss_variance:.6f}\n")
            f.write("Augmentation helped smooth training convergence.\n\n")
        elif aug_loss_variance > baseline_loss_variance:
            f.write(f"NO: {baseline_loss_variance:.6f} to {aug_loss_variance:.6f}\n")
            f.write("More data variation made training slightly noisier.\n\n")
        else:
            f.write("NO CHANGE in loss variance.\n\n")
        
        f.write("WHICH CLASSES IMPROVED THE MOST?\n")
        f.write("Per-class accuracy changes:\n")
        for i, cls in enumerate(class_names):
            change = per_class_improvements[i]
            sign = "+" if change >= 0 else ""
            f.write(f"  {cls}: {sign}{change:.4f} ({baseline_per_class[i]:.4f} -> {aug_per_class[i]:.4f})\n")
        f.write(f"\nBest improved: {best_improved_class} (+{np.max(per_class_improvements):.4f})\n")
        f.write(f"Worst affected: {worst_affected_class} ({np.min(per_class_improvements):.4f})\n\n")
        
        f.write("WHY AUGMENTATION CAN BE USEFUL FOR SATELLITE IMAGERY:\n")
        f.write("It can simulate camera angle changes without needing new data\n")
        
        f.write("\nSAVED FILES:\n")
        f.write(f"- Baseline confusion matrix: {baseline_cm_path}\n")
        f.write(f"- Augmented confusion matrix: {aug_cm_path}\n")
        f.write(f"- Training loss comparison: {loss_plot_path}\n")
    
    print(f"\nSaved augmentation analysis: {analysis_path}")
    print(f"Baseline: {baseline_acc:.4f} acc | Augmented: {aug_acc:.4f} acc (diff: {acc_diff:+.4f})")


# task 14
def train_more_epochs():
    from src.vision.image_dataset import EuroSATDataset
    BATCH_SIZE = 16
    LR = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_ds = EuroSATDataset(root_dir=Path("data/processed/images/train"), transform=transform)
    test_ds = EuroSATDataset(root_dir=Path("data/processed/images/test"), transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    class_names = train_ds.class_names

    def train_model_epochs(epochs, save_name):
        model = SimpleCNN(num_classes=len(class_names)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = torch.nn.CrossEntropyLoss()
        epoch_losses = []
        start = time.time()
        model.train()
        for e in range(epochs):
            tot = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                out = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward()
                opt.step()
                tot += loss.item()
            epoch_losses.append(tot / len(train_loader))
        train_time = time.time() - start

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.numpy())
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH.parent / save_name)
        out_dir = Path("reports"); out_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(6,6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="viridis", values_format="d", colorbar=False)
        plt.tight_layout()
        cm_path = out_dir / f"confusion_{save_name.replace('.pt','')}.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)

        return epoch_losses, train_time, acc, cm, cm_path

    losses_8, time_8, acc_8, cm_8, cm8_path = train_model_epochs(8, "cnn_8epochs.pt")
    losses_20, time_20, acc_20, cm_20, cm20_path = train_model_epochs(20, "cnn_20epochs.pt")

    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_more_epochs.txt", "w") as f:
        f.write("TRAIN MORE EPOCHS ANALYSIS\n")
        f.write("==========================\n\n")
        f.write(f"Baseline (8 epochs): final accuracy: {acc_8:.4f}, training time: {time_8:.2f}s, final loss: {losses_8[-1]:.4f}\n")
        f.write(f"Longer (20 epochs): final accuracy: {acc_20:.4f}, training time: {time_20:.2f}s, final loss: {losses_20[-1]:.4f}\n\n")
        f.write("DID ACCURACY IMPROVE?\n")
        diff = acc_20 - acc_8
        f.write(f"{diff:+.4f} (20 epochs vs 8 epochs)\n\n")
        f.write("DID LOSS CONTINUE TO DECREASE?\n")
        f.write(f"Loss 8 final: {losses_8[-1]:.4f}; Loss 20 final: {losses_20[-1]:.4f}\n\n")
        f.write("IS MORE TRAINING ALWAYS BETTER?\n")
        f.write("Not always - you must look for possible overfitting. Longer training may plateau or overfit.\n\n")
        f.write(f"Saved confusion matrices: {cm8_path}, {cm20_path}\n")

    print("Saved train_more_epochs.txt")


# task 15
def train_deeper_model():
    from src.vision.image_dataset import EuroSATDataset
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    train_ds = EuroSATDataset(root_dir=Path("data/processed/images/train"), transform=transform)
    test_ds = EuroSATDataset(root_dir=Path("data/processed/images/test"), transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    class_names = train_ds.class_names
    
    class ShallowCNN(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),  # 64->32
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)   # 32->16
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(32 * 16 * 16, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    def train_and_eval_model(model, name):
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = torch.nn.CrossEntropyLoss()
        epoch_losses = []
        start = time.time()
        model.train()
        for e in range(EPOCHS):
            tot = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                out = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward()
                opt.step()
                tot += loss.item()
            epoch_losses.append(tot / len(train_loader))
            print(f"{name} epoch {e+1}/{EPOCHS} loss {epoch_losses[-1]:.4f}")
        train_time = time.time() - start

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.numpy())
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        return epoch_losses, train_time, acc, cm

    shallow = ShallowCNN(num_classes=len(class_names))
    deeper = SimpleCNN(num_classes=len(class_names))

    shallow_losses, shallow_time, shallow_acc, shallow_cm = train_and_eval_model(shallow, "Shallow")
    deeper_losses, deeper_time, deeper_acc, deeper_cm = train_and_eval_model(deeper, "Deeper")

    out_dir = Path("reports"); out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=shallow_cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="viridis", values_format="d", colorbar=False)
    plt.tight_layout()
    shallow_cm_path = out_dir / "confusion_shallow.png"
    plt.savefig(shallow_cm_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=deeper_cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="viridis", values_format="d", colorbar=False)
    plt.tight_layout()
    deeper_cm_path = out_dir / "confusion_deeper.png"
    plt.savefig(deeper_cm_path, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "deeper_model_analysis.txt", "w") as f:
        f.write("DEEPER MODEL ANALYSIS\n")
        f.write("=====================\n\n")
        f.write(f"Shallow (2 convs): acc={shallow_acc:.4f}, time={shallow_time:.2f}s, final loss={shallow_losses[-1]:.4f}\n")
        f.write(f"Deeper (3 convs): acc={deeper_acc:.4f}, time={deeper_time:.2f}s, final loss={deeper_losses[-1]:.4f}\n\n")
        f.write("DID THE DEEPER MODEL IMPROVE PERFORMANCE?\n")
        f.write(f"Acc diff: {deeper_acc - shallow_acc:+.4f}\n\n")
        f.write("DID TRAINING BECOME SLOWER?\n")
        f.write(f"Time ratio deeper/shallow: {deeper_time / shallow_time:.2f}x\n\n")
        f.write("TRADE-OFF:\n")
        f.write("Deeper models may learn better/more features but require more compute and may overfit.\n")
        f.write(f"\nConfusion matrices saved: {shallow_cm_path}, {deeper_cm_path}\n")

    print("Saved deeper_model_analysis.txt")
    
    
def main():
    # prev_part()
    # prepare_conf_mx()
    # compare_cnn_vs_ml()
    # train_with_augmentation()
    # train_more_epochs()
    train_deeper_model()


if __name__ == "__main__":
    main()
