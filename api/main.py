from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from kafka import KafkaProducer
from pathlib import Path
import xgboost as xgb
import json
import os
import numpy as np


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Real-Time Fraud Detection System",
    description="Fraud detection using XGBoost, Kafka and FastAPI",
    version="1.0"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model.json"

UI_PATH = Path(__file__).resolve().parent / "index.html"


# ============================================================
# LOAD MODEL
# ============================================================

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

print("XGBoost model loaded successfully!")


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.80

KAFKA_SERVER = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka:9092"
)


# ============================================================
# KAFKA PRODUCER
# ============================================================

producer = None

try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    print(f"Kafka producer connected to: {KAFKA_SERVER}")

except Exception as e:
    print(f"Kafka producer connection failed: {e}")


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


# ============================================================
# TRANSACTION MODEL
# ============================================================

class Transaction(BaseModel):

    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float = Field(ge=0)


class BatchRequest(BaseModel):
    transactions: list[Transaction]


# ============================================================
# ROOT - WEB UI
# ============================================================

@app.get("/")
def home():
    return FileResponse(UI_PATH)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "kafka_server": KAFKA_SERVER
    }


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_transaction(transaction):

    values = [
        getattr(transaction, feature)
        for feature in FEATURES
    ]

    X = np.array([values])

    probability = float(
        model.predict_proba(X)[0][1]
    )

    if probability >= THRESHOLD:
        prediction = "FRAUD"
        risk = "HIGH"

    elif probability >= 0.50:
        prediction = "SUSPICIOUS"
        risk = "MEDIUM"

    else:
        prediction = "NORMAL"
        risk = "LOW"

    return probability, prediction, risk


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    probability, prediction, risk = predict_transaction(
        transaction
    )

    # Send transaction to Kafka
    kafka_status = "not_sent"

    if producer is not None:

        try:

            producer.send(
                "fraud-transactions",
                value=transaction.model_dump()
            )

            producer.flush()

            kafka_status = "sent"

        except Exception as e:

            kafka_status = f"failed: {str(e)}"

    return {

        "status": "success",

        "fraud_probability": round(
            probability,
            4
        ),

        "prediction": prediction,

        "risk_level": risk,

        "threshold": THRESHOLD,

        "kafka_status": kafka_status
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/predict_batch")
def predict_batch(request: BatchRequest):

    results = []

    fraud_count = 0
    suspicious_count = 0
    normal_count = 0

    for index, transaction in enumerate(
        request.transactions,
        start=1
    ):

        probability, prediction, risk = predict_transaction(
            transaction
        )

        if prediction == "FRAUD":
            fraud_count += 1

        elif prediction == "SUSPICIOUS":
            suspicious_count += 1

        else:
            normal_count += 1

        results.append({

            "transaction": index,

            "fraud_probability": round(
                probability,
                4
            ),

            "prediction": prediction,

            "risk_level": risk,

            "amount": transaction.Amount

        })

    total = len(results)

    fraud_rate = (
        (fraud_count / total) * 100
        if total > 0
        else 0
    )

    return {

        "status": "success",

        "total_transactions": total,

        "fraud_count": fraud_count,

        "suspicious_count": suspicious_count,

        "normal_count": normal_count,

        "fraud_rate": round(
            fraud_rate,
            2
        ),

        "results": results
    }