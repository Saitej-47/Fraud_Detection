
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from kafka import KafkaProducer
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
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
# PROMETHEUS METRICS
# ============================================================

Instrumentator().instrument(app).expose(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model.json"

API_DIR = Path(__file__).resolve().parent

UI_PATH = API_DIR / "index.html"

BACKGROUND_PATH = API_DIR / "background.png"


# ============================================================
# SERVE STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=str(API_DIR)),
    name="static"
)


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

# Kafka is optional.
# It will only be enabled when an external Kafka server
# is configured through the environment variable.

KAFKA_SERVER = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    ""
).strip()

if not KAFKA_SERVER:
    KAFKA_SERVER = None


# ============================================================
# KAFKA PRODUCER
# ============================================================

producer = None

if KAFKA_SERVER:

    try:

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda v:
                json.dumps(v).encode("utf-8")
        )

        print(
            f"Kafka producer initialized: {KAFKA_SERVER}"
        )

    except Exception as e:

        print(
            f"Kafka producer connection failed: {e}"
        )

else:

    print(
        "Kafka is disabled. Running without Kafka."
    )


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
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
# BACKGROUND IMAGE
# ============================================================

@app.get("/background.png")
def background():

    return FileResponse(
        BACKGROUND_PATH,
        media_type="image/png"
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "model_loaded":
            model is not None,

        "kafka_server":
            KAFKA_SERVER,

        "kafka_enabled":
            producer is not None,

        "threshold":
            THRESHOLD
    }


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_transaction(transaction):

    values = [
        getattr(
            transaction,
            feature
        )
        for feature in FEATURES
    ]

    X = np.array(
        [values]
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    # --------------------------------------------
    # FRAUD
    # --------------------------------------------

    if probability >= THRESHOLD:

        prediction = "FRAUD"

        risk = "HIGH"

    # --------------------------------------------
    # SUSPICIOUS
    # --------------------------------------------

    elif probability >= 0.50:

        prediction = "SUSPICIOUS"

        risk = "MEDIUM"

    # --------------------------------------------
    # NORMAL
    # --------------------------------------------

    else:

        prediction = "NORMAL"

        risk = "LOW"

    return (
        probability,
        prediction,
        risk
    )


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    transaction: Transaction
):

    probability, prediction, risk = \
        predict_transaction(transaction)

    # ========================================================
    # SEND TO KAFKA
    # ========================================================

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

            kafka_status = (
                f"failed: {str(e)}"
            )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "status": "success",

        "fraud_probability":
            round(
                probability,
                4
            ),

        "prediction":
            prediction,

        "risk_level":
            risk,

        "threshold":
            THRESHOLD,

        "kafka_status":
            kafka_status
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/predict_batch")
def predict_batch(
    request: BatchRequest
):

    results = []

    fraud_count = 0

    suspicious_count = 0

    normal_count = 0

    # ========================================================
    # PROCESS TRANSACTIONS
    # ========================================================

    for index, transaction in enumerate(
        request.transactions,
        start=1
    ):

        probability, prediction, risk = \
            predict_transaction(
                transaction
            )

        # --------------------------------------------
        # COUNTERS
        # --------------------------------------------

        if prediction == "FRAUD":

            fraud_count += 1

        elif prediction == "SUSPICIOUS":

            suspicious_count += 1

        else:

            normal_count += 1

        # --------------------------------------------
        # RESULT
        # --------------------------------------------

        results.append({

            "transaction":
                index,

            "fraud_probability":
                round(
                    probability,
                    4
                ),

            "prediction":
                prediction,

            "risk_level":
                risk,

            "amount":
                transaction.Amount

        })

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(results)

    fraud_rate = (

        (fraud_count / total) * 100

        if total > 0

        else 0
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "status":
            "success",

        "total_transactions":
            total,

        "fraud_count":
            fraud_count,

        "suspicious_count":
            suspicious_count,

        "normal_count":
            normal_count,

        "fraud_rate":
            round(
                fraud_rate,
                2
            ),

        "results":
            results
    }
