"""Clean the tourism dataset and save training/testing splits."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# Exclude identifiers and information collected during sales contact.
DROP_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "TypeofContact",
    "DurationOfPitch",
    "NumberOfFollowups",
    "ProductPitched",
    "PitchSatisfactionScore",
]


def main():
    df = pd.read_csv(DATA_PATH)
    original_rows = len(df)

    # Remove extra spaces from text values.
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    # Correct a known spelling inconsistency.
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Remove identifiers and sales-interaction information.
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Keep separate customer records, even when their profiles match.
    X = df.drop(columns="ProdTaken")
    y = df["ProdTaken"].astype(int)

    # Group identical predictor profiles so they cannot appear
    # in both training and testing data.
    groups = pd.util.hash_pandas_object(X, index=False)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    Xtrain = X.iloc[train_idx].copy()
    Xtest = X.iloc[test_idx].copy()
    ytrain = y.iloc[train_idx].copy()
    ytest = y.iloc[test_idx].copy()

    # Check that both classes appear in each split.
    if ytrain.nunique() != 2 or ytest.nunique() != 2:
        raise ValueError("Both splits must contain both target classes.")

    # Verify that identical profiles do not cross the split boundary.
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:
        raise ValueError("Matching profiles appear in both splits.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Xtrain.to_csv(OUTPUT_DIR / "Xtrain.csv", index=False)
    Xtest.to_csv(OUTPUT_DIR / "Xtest.csv", index=False)
    ytrain.to_csv(OUTPUT_DIR / "ytrain.csv", index=False)
    ytest.to_csv(OUTPUT_DIR / "ytest.csv", index=False)

    print("DATA PREPARATION SUCCESSFUL")
    print(f"Original records: {original_rows:,}")
    print(f"Predictors retained: {X.shape[1]}")
    print(f"Training shape: {Xtrain.shape}")
    print(f"Testing shape: {Xtest.shape}")
    print(f"Training purchase rate: {ytrain.mean():.2%}")
    print(f"Testing purchase rate: {ytest.mean():.2%}")
    print(f"Identical profiles shared between splits: {len(overlap)}")

    print("\nRetained predictors:")
    for column in X.columns:
        print(f"- {column}")

    print("\nSaved files:")
    for name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
