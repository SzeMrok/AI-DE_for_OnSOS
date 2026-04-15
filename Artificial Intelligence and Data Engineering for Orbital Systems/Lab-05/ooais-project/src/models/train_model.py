from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import csv


# task 1

dsfp = Path("data/processed/model_features.csv")

if not dsfp.exists() or not dsfp.is_file():
    raise FileNotFoundError(f"Dataset file not found: {dsfp}")
dsf = open(dsfp, "r")
dataset = csv.DictReader(dsf)

records = []
for row in dataset:
    records.append(row)

print(f"=== Machine Learning: Loading Feature Dataset ===\nInput file: {dsfp}\nRecords loaded: {len(records)}\nColumns: {dataset.fieldnames}")

# task 2

X: list[list[float]] = []
for record in records:
    feature = []
    for n in dict(record).values():
        feature.append(n)
    X.append(feature)

lfp = Path("data/processed/model_labels.csv")

if not lfp.exists() or not lfp.is_file():
    raise FileNotFoundError(f"Dataset file not found: {lfp}")
lf = open(lfp, "r")
labels = csv.DictReader(lf)

y = []
target_vals = []
for label in labels:
    n = label.get(labels.fieldnames[0])
    y.append(n)
    if n not in target_vals:
        target_vals.append(n)
    

print(f"\n=== Machine Learning: Preparing Features and Target ===\nNumber of samples in X: {len(X)}\nNumber of labels in y: {len(y)}\nTarget values detected: {target_vals}")

# task 3

train_split = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n=== Machine Learning: Train/Test Split ===\nTraining samples: {len(train_split[0])}\nTest samples: {len(train_split[1])}")

# task 4

model = DecisionTreeClassifier()
model.fit(train_split[0], train_split[2])

print(f"\n=== Machine Learning: Model Training ===\nModel: {model}\nTraining completed successfully.")

# task 5

predictions = model.predict(train_split[1])

print(f"\n=== Machine Learning: Prediction ===\nPredictions generated for test set.\nNumber of predictions: {len(predictions)}\nExample predictions:\n{[predictions.item(0), predictions.item(1), predictions.item(2), predictions.item(3), predictions.item(4)]}")

# task 6

score = accuracy_score(train_split[3], predictions)
matrix = confusion_matrix(train_split[3], predictions)

print(f"\n=== Machine Learning: Evaluation ===\nAccuracy: {score:.3f}\nConfusion Matrix:\n{matrix}")

# task 7

