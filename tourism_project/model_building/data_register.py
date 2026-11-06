"""Validate and summarize the tourism dataset."""

from pathlib import Path
import pandas as pd

# Resolve the dataset path relative to this script.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The dataset is empty.")

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    if df["ProdTaken"].isna().any():
        raise ValueError("The target column contains missing values.")

    if set(df["ProdTaken"].unique()) != {0, 1}:
        raise ValueError("ProdTaken must contain both classes: 0 and 1.")

    if df["CustomerID"].isna().any():
        raise ValueError("CustomerID contains missing values.")

    if df["CustomerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs were found.")

    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    print("DATA REGISTRATION SUCCESSFUL")
    print(f"Dataset: {DATA_PATH.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Required columns verified: {len(EXPECTED_COLUMNS)}")
    print(f"Additional columns: {extra_columns}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate customer IDs: {df['CustomerID'].duplicated().sum()}")

    print("\nTarget distribution:")
    print(df["ProdTaken"].value_counts().sort_index().to_string())
    print(f"\nPurchase rate: {df['ProdTaken'].mean():.2%}")


if __name__ == "__main__":
    main()
