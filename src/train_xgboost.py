import pandas as pd
import joblib
import mlflow
import mlflow.xgboost

from pathlib import Path

from xgboost import XGBClassifier

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
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model.json"
MLFLOW_DB = BASE_DIR / "mlflow.db"


# ==========================================
# MLflow Setup
# ==========================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    "Fraud_Detection_XGBoost"
)


# ==========================================
# Model Parameters
# ==========================================

n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
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
    # 4. Calculate Class Imbalance
    # ======================================

    normal_count = (y_train == 0).sum()
    fraud_count = (y_train == 1).sum()

    scale_pos_weight = normal_count / fraud_count

    print("\nNormal transactions:", normal_count)
    print("Fraud transactions:", fraud_count)
    print("Scale Pos Weight:", scale_pos_weight)


    # ======================================
    # 5. Create XGBoost Model
    # ======================================

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1
    )


    # ======================================
    # 6. Log Parameters
    # ======================================

    mlflow.log_param(
        "model",
        "XGBoost"
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "learning_rate",
        learning_rate
    )

    mlflow.log_param(
        "subsample",
        subsample
    )

    mlflow.log_param(
        "colsample_bytree",
        colsample_bytree
    )

    mlflow.log_param(
        "scale_pos_weight",
        scale_pos_weight
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

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")


    # ======================================
    # 8. Predictions
    # ======================================

    # Probability of fraud
    probabilities = model.predict_proba(X_test)[:, 1]

    # Default threshold
    predictions = (
        probabilities >= 0.5
    ).astype(int)


    # ======================================
    # 9. Calculate Metrics
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
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )


    # ======================================
    # 10. Confusion Matrix
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
    # 11. Classification Report
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
    # 12. Print Metrics
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
    # 13. Log Metrics to MLflow
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
    # 14. Save Model
    # ======================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_model(
        MODEL_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)


    # ======================================
    # 15. Log Model to MLflow
    # ======================================

    mlflow.xgboost.log_model(
        model,
        name="xgboost_fraud_model"
    )

    print("\nModel logged to MLflow.")


    # ======================================
    # 16. Run Information
    # ======================================

    run_id = mlflow.active_run().info.run_id

    print("\n==============================")
    print("MLFLOW RUN")
    print("==============================")

    print("MLflow Run ID:")
    print(run_id)

    print("\nExperiment:")
    print("Fraud_Detection_XGBoost")

    print("\nTraining completed successfully!")