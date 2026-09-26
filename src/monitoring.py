from pathlib import Path
import pandas as pd
import subprocess
import sys

from evidently import Report
from evidently.presets import DataDriftPreset


# ==========================================
# Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "logs" / "predictions.csv"
REPORT_FILE = BASE_DIR / "logs" / "drift_report.html"

TRAIN_SCRIPT = BASE_DIR / "src" / "train_xgboost.py"


# ==========================================
# Drift threshold
# ==========================================

# Retraining will happen only when
# 50% or more of the columns are drifted.

DRIFT_THRESHOLD = 0.50


# ==========================================
# Check prediction data
# ==========================================

if not DATA_FILE.exists():

    print("No prediction data found.")
    print("Run kafka_consumer.py first.")
    sys.exit()


df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df)} transactions.")


# ==========================================
# Model features
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
# Check data size
# ==========================================

if len(df) < 20:

    print("Not enough transactions for reliable drift detection.")
    print(f"Current transactions: {len(df)}")
    print("Need at least 20 transactions.")

    sys.exit()


# ==========================================
# Split reference and current data
# ==========================================

split = len(df) // 2

reference_data = df[features].iloc[:split]

current_data = df[features].iloc[split:]


print(f"Reference transactions: {len(reference_data)}")
print(f"Current transactions:   {len(current_data)}")


# ==========================================
# Run Evidently
# ==========================================

report = Report(
    [
        DataDriftPreset()
    ]
)

result = report.run(
    current_data,
    reference_data
)


# ==========================================
# Save HTML report
# ==========================================

result.save_html(str(REPORT_FILE))

print("\n================================")
print("EVIDENTLY MONITORING COMPLETE")
print("================================")

print("Report saved to:")
print(REPORT_FILE)


# ==========================================
# Extract drift percentage
# ==========================================

report_dict = result.dict()


def find_drift_share(obj):
    """
    Recursively search Evidently's result
    for the share of drifted columns.
    """

    if isinstance(obj, dict):

        for key, value in obj.items():

            key_lower = str(key).lower()

            # Look for keys related to
            # share + drift

            if (
                "share" in key_lower
                and "drift" in key_lower
                and isinstance(value, (int, float))
            ):

                return float(value)

            # Search deeper

            found = find_drift_share(value)

            if found is not None:
                return found


    elif isinstance(obj, list):

        for item in obj:

            found = find_drift_share(item)

            if found is not None:
                return found


    return None


drift_share = find_drift_share(report_dict)


# ==========================================
# Drift decision
# ==========================================

if drift_share is None:

    print("\nCould not automatically extract drift percentage.")

    print("The HTML report was generated successfully.")

    print("\nEvidently metrics:")

    for metric in report_dict.get("metrics", []):

        print(
            metric.get("metric_name"),
            "->",
            metric.get("value")
        )

    sys.exit()


print(f"\nDrifted column share: {drift_share:.2%}")
print(f"Retraining threshold: {DRIFT_THRESHOLD:.2%}")


# ==========================================
# Retraining decision
# ==========================================

if drift_share >= DRIFT_THRESHOLD:

    print("\n⚠️ DATA DRIFT DETECTED")

    print("Starting XGBoost retraining...")

    subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=str(BASE_DIR),
        check=True
    )

    print("\n✅ MODEL RETRAINING COMPLETED")


else:

    print("\n✅ No significant dataset drift.")

    print("Model retraining is not required.")