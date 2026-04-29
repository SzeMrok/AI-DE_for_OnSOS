from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import pandas as pd



# task 1

def validate_input_files() -> None:
    mffp = Path("data/processed/model_features.csv")
    mlfp = Path("data/processed/model_labels.csv")
    
    mff = mffp.exists()
    mlf = mffp.exists()
    
    if not (mff and mlf):
        raise SystemExit(
            "Error: missing required input file(s): " 
            + ('\n- ' + mffp) if mff else "" 
            + ('\n- ' + mlfp) if mlf else ""
        )
        
# task 2

def load_data():
    mffp = Path("data/processed/model_features.csv")
    mlfp = Path("data/processed/model_labels.csv")
    features_df = pd.read_csv(mffp)
    labels_df = pd.read_csv(mlfp)
    
    print(
        f"=== Model Playground: Loading Data ===\n"
        f"Feature file: {mffp}\n"
        f"Label file: {mlfp}\n"
    )
    
    return features_df, labels_df

# task 3

def inspect_data(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    if features_df.empty or labels_df.empty:
        raise Exception(
            "Empty DataFrame(s) provided:" 
            + "\n- features dataset" if features_df.empty else "" 
            + "\n- labels dataset" if labels_df.empty else ""
        )
    
    if len(features_df) != len(labels_df):
        raise Exception("Different DataFrame lengths.")
    
    if "anomaly_flag" not in labels_df.columns:
        raise Exception("Missing \"anomaly_flag\" column in labels dataset.")
    
    
    target_vals = sorted(labels_df["anomaly_flag"].unique().tolist())
    feature_cols = list(features_df.columns)

    print(
        f"=== Model Playground: Data Inspection ===\n"
        f"Number of samples: {len(features_df)}\n"
        f"Number of features: {len(feature_cols)}\n"
        f"Feature columns: {feature_cols}\n"
        f"Target values detected: {target_vals}\n"
    )
    
# task 4

def prepare_features_and_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    X = features_df.values
    y = labels_df["anomaly_flag"].astype(int).values
    
    print(
        f"=== Model Playground: Preparing Features and Labels ===\n"
        f"X shape: {X.shape}\n"
        f"y shape: {y.shape}\n"
    )
    
    return X, y

# task 5

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    print(
        f"=== Model Playground: Train/Test Split ===\n"
        f"Training samples: {len(X_train)}\n"
        f"Testing samples: {len(X_test)}\n"
    )
    
    return X_train, X_test, y_train, y_test

# task 6

def define_model() -> dict[str, Any]:
    models = {
        "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    return models

# task 7

def train_models(models: dict[str, Any], X_train, y_train) -> dict[str, Any]:
    trained_models = {}
    print("=== Model Playground: Training Models ===")
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{model_name}: trained")
        trained_models[model_name] = model
    
    print("")
    return trained_models

# task 8

def generate_predictions(trained_models: dict[str, Any], X_test) -> list[dict]:
    results: list[dict] = []
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        result = {
            "name": model_name,
            "model": model,
            "y_pred": y_pred
        }
        
        results.append(result)
        
    return results

# task 9

def print_example_predictions(prediction_results: list[dict], y_test, num_examples=5):
    print("=== Model Playground: Example Predictions ===")
    for i in range(num_examples):
        line = f"True: {y_test[i]}"
        
        for result in prediction_results:
            model_name = result["name"]
            y_pred = result["y_pred"]
            line += f" | {model_name}: {y_pred[i]}"
            
        print(line)
    print("")
    
# task 10

def compute_accuracy(prediction_results: list[dict], y_test) -> list[dict]:
    print("=== Model Playground: Accuracy Comparison ===")
    for result in prediction_results:
        y_pred = result["y_pred"]
        accuracy = accuracy_score(y_test, y_pred)
        result["accuracy"] = accuracy
        print(f"{result['name']}: {accuracy:.4f}")
    
    print("")        
    return prediction_results

# task 11

def compute_detailed_metrics(prediction_results: list[dict], y_test, do_print = True) -> list[dict]:
    print("=== Model Playground: Detailed Evaluation ===")
    for result in prediction_results:
        y_pred = result["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        result["confusion_matrix"] = cm
        result["classification_report"] = report
        if do_print:
            print(f"Model: {result['name']}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print("\nConfusion Matrix:")
            print(result["confusion_matrix"])
            print("\nClass labels:")
            print("0 -> normal observation")
            print("1 -> anomaly")
            print("\nClassification Report:")
            print("------------------------------------------------------------")
            print("Class Precision Recall F1-score Support")
            print("------------------------------------------------------------")
            print(
                f"0 (normal) "
                f"{report['0']['precision']:.2f} "
                f"{report['0']['recall']:.2f} "
                f"{report['0']['f1-score']:.2f} "
                f"{int(report['0']['support'])}"
            )
            print(
                f"1 (anomaly) "
                f"{report['1']['precision']:.2f} "
                f"{report['1']['recall']:.2f} "
                f"{report['1']['f1-score']:.2f} "
                f"{int(report['1']['support'])}"
            )
            print("------------------------------------------------------------")
            print(
                f"Macro average "
                f"{report['macro avg']['precision']:.2f} "
                f"{report['macro avg']['recall']:.2f} "
                f"{report['macro avg']['f1-score']:.2f} "
                f"{int(report['macro avg']['support'])}"
            )
            print(
                f"Weighted average "
                f"{report['weighted avg']['precision']:.2f} "
                f"{report['weighted avg']['recall']:.2f} "
                f"{report['weighted avg']['f1-score']:.2f} "
                f"{int(report['weighted avg']['support'])}"
            )
    if not do_print:
        print("Report printing skipped.")
    return prediction_results

# task 12

def rank_models(evaluation_results):
    print("\n=== Model Playground: Ranking ===")
    sorted_results = sorted(
        evaluation_results,
        key=lambda result: result["accuracy"],
        reverse=True
    )
    for index, result in enumerate(sorted_results, start=1):
        print(f"{index}. {result['name']} - {result['accuracy']:.4f}")
        
    return sorted_results

# task 13

def define_mod_models() -> dict[str, Any]:
    print("\n=== Model Playground: Controlled Experiments ===\n")
    experimental_models = {}
    for depth in [2, 3, 5]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        experimental_models[f"Decision Tree (max depth={depth})"] = model
        
    for n_estimators in [5, 10, 50]:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        experimental_models[f"Random Forest (n_estimators={n_estimators})"] = model
        
    return experimental_models

def plot_model_comparison(model_results: list[dict]) -> None:
    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tree_depths = []
    tree_accuracies = []
    forest_trees = []
    forest_accuracies = []

    for result in model_results:
        name = result["name"]
        if "max depth=" in name:
            tree_depths.append(int(name.split("max depth=")[1].rstrip(")")))
            tree_accuracies.append(result["accuracy"])
        elif "n_estimators=" in name:
            forest_trees.append(int(name.split("n_estimators=")[1].rstrip(")")))
            forest_accuracies.append(result["accuracy"])

    if tree_depths:
        plt.figure(figsize=(8, 5))
        plt.plot(tree_depths, tree_accuracies, marker="o")
        plt.xlabel("Depth")
        plt.ylabel("Accuracy")
        plt.title("Decision Tree accuracy vs depth")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "decision_tree_accuracy_vs_depth.png", dpi=150)
        plt.close()

    if forest_trees:
        plt.figure(figsize=(8, 5))
        plt.plot(forest_trees, forest_accuracies, marker="o")
        plt.xlabel("Number of trees")
        plt.ylabel("Accuracy")
        plt.title("Random Forest accuracy vs number of trees")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "random_forest_accuracy_vs_trees.png", dpi=150)
        plt.close()




validate_input_files()
fdf, ldf = load_data()
inspect_data(fdf, ldf)
X, y = prepare_features_and_labels(fdf, ldf)
X_train, X_test, y_train, y_test = split_data(X, y)  
models = define_model()
tr_models = train_models(models, X_train, y_train)
pred_results = generate_predictions(tr_models, X_test)
print_example_predictions(pred_results, y_test)
pred_results = compute_accuracy(pred_results, y_test)
pred_results = compute_detailed_metrics(pred_results, y_test, do_print=False)
sorted_results = rank_models(pred_results)
experiment_models = define_mod_models()
tr_mod_models = train_models(experiment_models, X_train, y_train)
mod_pred_results = generate_predictions(tr_mod_models, X_test)
mod_pred_results = compute_accuracy(mod_pred_results, y_test)
plot_model_comparison(mod_pred_results)



    