from kafka import KafkaProducer
import pandas as pd
import json
from pathlib import Path


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "raw" / "creditcard.csv"


# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded: {len(df)} transactions")


# ==========================================
# Get a REAL fraud transaction
# ==========================================

fraud_transaction = df[df["Class"] == 1].iloc[0]


print("\nReal fraud transaction selected!")
print(f"Dataset row index: {fraud_transaction.name}")


# ==========================================
# Features
# ==========================================

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


# ==========================================
# Create transaction
# ==========================================

transaction = {
    feature: float(fraud_transaction[feature])
    for feature in features
}


# ==========================================
# Kafka Producer
# ==========================================

producer = KafkaProducer(
    bootstrap_servers="localhost:29092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)


# ==========================================
# Send fraud transaction
# ==========================================

producer.send(
    "fraud-transactions",
    value=transaction
)

producer.flush()
producer.close()


print("\n================================")
print("REAL FRAUD TRANSACTION SENT")
print("================================")

print(f"Actual dataset Class: {int(fraud_transaction['Class'])}")
print(f"Amount: {fraud_transaction['Amount']}")