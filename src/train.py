import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "isolation_forest.pkl"
MLFLOW_DB = BASE_DIR / "mlflow.db"


# ==========================================
# MLflow Setup
# ==========================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    "Fraud_Detection_Isolation_Forest"
)


# ==========================================
# Model Parameters
# ==========================================

n_estimators = 100

# Get contamination from command line
# Default = 0.002
contamination = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002

random_state = 42


# ==========================================
# Start MLflow Run
# ==========================================

with mlflow.start_run():

    # ======================================
    # 1. Load Dataset
    # ======================================

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)


    # ======================================
    # 2. Separate Features and Target
    # ======================================

    X = df.drop("Class", axis=1)
    y = df["Class"]


    # ======================================
    # 3. Train/Test Split
    # ======================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))


    # ======================================
    # 4. Display Parameters
    # ======================================

    print("\n==============================")
    print("MODEL PARAMETERS")
    print("==============================")

    print("Model:", "IsolationForest")
    print("n_estimators:", n_estimators)
    print("contamination:", contamination)
    print("random_state:", random_state)


    # ======================================
    # 5. Create Isolation Forest
    # ======================================

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )


    # ======================================
    # 6. Log Parameters to MLflow
    # ======================================

    mlflow.log_param(
        "model",
        "IsolationForest"
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "contamination",
        contamination
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "test_size",
        0.20
    )


    # ======================================
    # 7. Train Model
    # ======================================

    print("\nTraining Isolation Forest...")

    model.fit(X_train)

    print("Training completed.")


    # ======================================
    # 8. Make Predictions
    # ======================================

    predictions = model.predict(X_test)

    # Isolation Forest output:
    #
    #  1  = Normal
    # -1  = Anomaly
    #
    # Convert to:
    #
    #  0  = Normal
    #  1  = Fraud

    predictions = [
        1 if prediction == -1 else 0
        for prediction in predictions
    ]


    # ======================================
    # 9. Calculate Anomaly Scores
    # ======================================

    # Lower decision_function values
    # indicate more anomalous transactions.
    #
    # We multiply by -1 so that:
    #
    # Higher score = more anomalous
    #
    # This is useful for ROC-AUC and PR-AUC.

    anomaly_scores = -model.decision_function(X_test)


    # ======================================
    # 10. Calculate Metrics
    # ======================================

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

    roc_auc = roc_auc_score(
        y_test,
        anomaly_scores
    )

    pr_auc = average_precision_score(
        y_test,
        anomaly_scores
    )


    # ======================================
    # 11. Confusion Matrix
    # ======================================

    print("\n==============================")
    print("CONFUSION MATRIX")
    print("==============================")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    # ======================================
    # 12. Classification Report
    # ======================================

    print("\n==============================")
    print("CLASSIFICATION REPORT")
    print("==============================")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Fraud"
            ],
            zero_division=0
        )
    )


    # ======================================
    # 13. Print Metrics
    # ======================================

    print("==============================")
    print("MODEL METRICS")
    print("==============================")

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)


    # ======================================
    # 14. Log Metrics to MLflow
    # ======================================

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.log_metric(
        "pr_auc",
        pr_auc
    )


    # ======================================
    # 15. Save Model Locally
    # ======================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)


    # ======================================
    # 16. Log Model to MLflow
    # ======================================

    mlflow.sklearn.log_model(
        model,
        name="isolation_forest_model"
    )

    print("\nModel logged to MLflow.")


    # ======================================
    # 17. Display MLflow Run ID
    # ======================================

    run_id = mlflow.active_run().info.run_id

    print("\n==============================")
    print("MLFLOW RUN")
    print("==============================")

    print("MLflow Run ID:")
    print(run_id)

    print("\nExperiment:")
    print("Fraud_Detection_Isolation_Forest")

    print("\nTraining completed successfully!")