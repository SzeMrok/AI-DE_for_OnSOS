from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
import csv
import joblib


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

print(f"\n=== Machine Learning: Model Training ===\nModel: {model.__class__.__name__}\nTraining completed successfully.")

# task 5

predictions = model.predict(train_split[1])

print(f"\n=== Machine Learning: Prediction ===\nPredictions generated for test set.\nNumber of predictions: {len(predictions)}\nExample predictions:\n{[predictions.item(0), predictions.item(1), predictions.item(2), predictions.item(3), predictions.item(4)]}")

# task 6

score = accuracy_score(train_split[3], predictions)
matrix = confusion_matrix(train_split[3], predictions)

print(f"\n=== Machine Learning: Evaluation ===\nAccuracy: {score:.3f}\nConfusion Matrix:\n{matrix}")

# task 7
model_path = Path("results/decision_tree_model.joblib")
joblib.dump(model, model_path)

tree_rules = export_text(model, feature_names=dataset.fieldnames)

print(f"=== Machine Learning: Saving and Inspecting Model ===\nSaved model: {model_path}\nModel type: {model.__class__.__name__}\nTree depth: {model.get_depth()}\nNumber of leaves: {model.get_n_leaves()}\n")

print(tree_rules)

# task 8

mep = Path("results/model_evaluation.txt")

with open(mep, "w") as f:
    f.write(f"OOAIS Model Evaluation\n======================\n\nModel: {model.__class__.__name__}\nTraining samples: {len(train_split[0])}\nTest samples: {len(train_split[1])}\n\nAccuracy: {score:.3f}\n\nConfusion Matrix:\n{matrix}")

print(f"=== Machine Learning: Saving Evaluation Results ===\nSaved file: {mep}\n")

# task 9

mtsp = Path("reports/model_training_summary.txt")

with open(mtsp, "w") as f:
    f.write(f"OOAIS Model Training Summary\n============================\n\nInput datasets\n--------------\n{dsfp}\n{lfp}\n\nDataset statistics\n------------------\nNumber of samples: {len(X)}\nNumber of features: {len(dataset.fieldnames)}\n\nModel\n-----\n{model.__class__.__name__}\n\nTrain/Test split\n----------------\nTraining samples: {len(train_split[0])}\nTest samples: {len(train_split[1])}\n\nEvaluation summary\n------------------\nAccuracy: {score:.3f}\nConfusion Matrix:\n{matrix}")

print(f"=== Machine Learning: Saving Training Report ===\nSaved file: {mtsp}")
