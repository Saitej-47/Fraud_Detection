import pandas as pd
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    print("Dataset Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nClass Distribution:")
    print(df["Class"].value_counts())

    return df


if __name__ == "__main__":
    load_data()