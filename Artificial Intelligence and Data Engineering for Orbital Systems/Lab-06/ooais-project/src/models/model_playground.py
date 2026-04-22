from pathlib import Path
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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

def train_models(models: dict[str, Any], X_train, y_train):
    for model_name, model in models.items():
        model



validate_input_files()
fdf, ldf = load_data()
inspect_data(fdf, ldf)
X, y = prepare_features_and_labels(fdf, ldf)
X_train, X_test, y_train, y_test = split_data(X, y)  
models = define_model()

    