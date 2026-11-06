"""Tune, evaluate, track, and save the tourism prediction model."""

from pathlib import Path
import json
import joblib
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "deployment"
REPORT_DIR = PROJECT_DIR / "reports"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the files produced by data preparation.
    Xtrain = pd.read_csv(DATA_DIR / "Xtrain.csv")
    Xtest = pd.read_csv(DATA_DIR / "Xtest.csv")
    ytrain = pd.read_csv(DATA_DIR / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(DATA_DIR / "ytest.csv").squeeze("columns")

    categorical = Xtrain.select_dtypes(include="object").columns.tolist()
    numeric = Xtrain.select_dtypes(exclude="object").columns.tolist()

    # Learned preprocessing is fitted within each training fold.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestClassifier(
            random_state=42,
            n_jobs=1,
        )),
    ])

    parameter_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 12],
        "model__min_samples_leaf": [1, 3],
        "model__class_weight": ["balanced"],
    }

    groups = pd.util.hash_pandas_object(Xtrain, index=False)

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring={
            "average_precision": "average_precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )

    print("Starting model tuning...", flush=True)
    search.fit(Xtrain, ytrain, groups=groups)

    # Save all cross-validation results.
    results = pd.DataFrame(search.cv_results_)
    results_path = REPORT_DIR / "cv_results.csv"
    results.to_csv(results_path, index=False)

    # Track experiments locally. No ngrok connection is required.
    tracking_dir = PROJECT_DIR / "mlruns"
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("Tourism_Purchase_Prediction")

    # Record every parameter combination and its validation results.
    for index, row in results.iterrows():
        with mlflow.start_run(run_name=f"candidate_{index + 1}"):
            mlflow.log_params(row["params"])
            mlflow.log_metrics({
                "cv_average_precision": float(
                    row["mean_test_average_precision"]
                ),
                "cv_recall": float(row["mean_test_recall"]),
                "cv_f1": float(row["mean_test_f1"]),
                "cv_roc_auc": float(row["mean_test_roc_auc"]),
            })

    # GridSearchCV refits the selected pipeline on all training data.
    best_model = search.best_estimator_

    # Evaluate the selected model on the untouched test set.
    predictions = best_model.predict(Xtest)
    probabilities = best_model.predict_proba(Xtest)[:, 1]

    metrics = {
        "test_accuracy": accuracy_score(ytest, predictions),
        "test_precision": precision_score(
            ytest, predictions, zero_division=0
        ),
        "test_recall": recall_score(
            ytest, predictions, zero_division=0
        ),
        "test_f1": f1_score(ytest, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(ytest, probabilities),
        "test_average_precision": average_precision_score(
            ytest, probabilities
        ),
    }
    metrics = {name: float(value) for name, value in metrics.items()}

    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(best_model, model_path)

    report = classification_report(
        ytest, predictions, zero_division=0
    )
    matrix = confusion_matrix(ytest, predictions, labels=[0, 1])

    (REPORT_DIR / "classification_report.txt").write_text(report)
    pd.DataFrame(
        matrix,
        index=["Actual_No", "Actual_Yes"],
        columns=["Predicted_No", "Predicted_Yes"],
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    with open(REPORT_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    with open(REPORT_DIR / "best_parameters.json", "w") as file:
        json.dump(search.best_params_, file, indent=2)

    with mlflow.start_run(run_name="selected_model"):
        mlflow.log_params(search.best_params_)
        mlflow.log_param("prediction_threshold", 0.5)
        mlflow.log_metric(
            "best_cv_average_precision", float(search.best_score_)
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifacts(str(REPORT_DIR), artifact_path="reports")

    print("\nMODEL TRAINING SUCCESSFUL")
    print("\nBest parameters:")
    print(json.dumps(search.best_params_, indent=2))

    print(f"\nBest CV average precision: {search.best_score_:.4f}")
    print(
        "Test positive-class proportion "
        f"(reference for average precision): {ytest.mean():.4f}"
    )

    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(report)

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)

    print(f"\nSaved model: {model_path}")
    print(f"Experiment tracking: {tracking_dir}")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
