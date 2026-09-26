from kafka import KafkaConsumer
import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime
import pandas as pd


# ==============================
# Load trained XGBoost model
# ==============================

model = xgb.XGBClassifier()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model.json"

print(f"Loading model from: {MODEL_PATH}")

model.load_model(str(MODEL_PATH))

print("XGBoost model loaded successfully!")


# ==============================
# Settings
# ==============================

THRESHOLD = 0.80

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PREDICTION_FILE = LOG_DIR / "predictions.csv"


# ==============================
# Kafka Consumer
# ==============================

consumer = KafkaConsumer(
    "fraud-transactions",
    bootstrap_servers="kafka:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="fraud-detection-group-v4",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)


print("\nFraud Detection Consumer Started...")
print("Waiting for transactions...\n")


# ==============================
# Feature order
# ==============================

features = [
    "Time",
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


# ==============================
# Process transactions
# ==============================

for message in consumer:

    transaction = message.value

    print("--------------------------------")
    print("Transaction received")
    print("--------------------------------")


    # Convert transaction into model input

    X = np.array([
        [transaction[feature] for feature in features]
    ])


    # ==============================
    # Prediction
    # ==============================

    probability = model.predict_proba(X)[0][1]


    if probability >= THRESHOLD:
        prediction = "FRAUD"
        risk = "HIGH"

    elif probability >= 0.50:
        prediction = "SUSPICIOUS"
        risk = "MEDIUM"

    else:
        prediction = "NORMAL"
        risk = "LOW"


    # ==============================
    # Display prediction
    # ==============================

    print(f"Fraud Probability : {probability:.4f}")
    print(f"Prediction        : {prediction}")
    print(f"Risk Level        : {risk}")


    # ==============================
    # Save prediction log
    # ==============================

    log_data = transaction.copy()

    log_data["fraud_probability"] = probability
    log_data["prediction"] = prediction
    log_data["risk_level"] = risk
    log_data["timestamp"] = datetime.now().isoformat()


    log_df = pd.DataFrame([log_data])


    # Create file if it doesn't exist
    if PREDICTION_FILE.exists():

        log_df.to_csv(
            PREDICTION_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        log_df.to_csv(
            PREDICTION_FILE,
            index=False
        )


    print(f"Prediction saved to: {PREDICTION_FILE}")
    print("--------------------------------\n")