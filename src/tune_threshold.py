import pandas as pd
import mlflow

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model.json"
MLFLOW_DB = BASE_DIR / "mlflow.db"


# ==========================================
# MLflow Setup
# ==========================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    "Fraud_Detection_Threshold_Tuning"
)


# ==========================================
# Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]


# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# Load XGBoost Model
# ==========================================

print("Loading XGBoost model...")

model = XGBClassifier()

model.load_model(MODEL_PATH)


# ==========================================
# Get Fraud Probabilities
# ==========================================

print("Generating fraud probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]


# ==========================================
# Thresholds
# ==========================================

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


# ==========================================
# Find Best Threshold
# ==========================================

best_threshold = 0
best_f1 = 0

results = []


print("\n==============================================")
print("THRESHOLD ANALYSIS")
print("==============================================")

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.3f}"
        f"{recall:<12.3f}"
        f"{f1:<12.3f}"
    )


    results.append(
        (
            threshold,
            precision,
            recall,
            f1
        )
    )


    # Select best threshold based on F1

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


# ==========================================
# Print Best Threshold
# ==========================================

print("\n==============================================")
print("BEST THRESHOLD")
print("==============================================")

print("Best Threshold:", best_threshold)
print("Best F1 Score:", best_f1)


# ==========================================
# Log Best Threshold to MLflow
# ==========================================

with mlflow.start_run():

    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "optimization_metric",
        "F1"
    )

    mlflow.log_param(
        "number_of_thresholds_tested",
        len(thresholds)
    )

    mlflow.log_param(
        "test_size",
        0.20
    )

    mlflow.log_metric(
        "best_threshold",
        best_threshold
    )

    mlflow.log_metric(
        "best_f1",
        best_f1
    )


    # Log every threshold result

    for threshold, precision, recall, f1 in results:

        threshold_name = str(
            threshold
        ).replace(".", "_")


        mlflow.log_metric(
            f"precision_{threshold_name}",
            precision
        )

        mlflow.log_metric(
            f"recall_{threshold_name}",
            recall
        )

        mlflow.log_metric(
            f"f1_{threshold_name}",
            f1
        )


    print("\nThreshold experiment logged to MLflow.")

    print(
        "MLflow Run ID:",
        mlflow.active_run().info.run_id
    )