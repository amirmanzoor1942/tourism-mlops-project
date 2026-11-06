# Tourism Package Purchase Prediction

Predict customer purchase propensity using a Random Forest pipeline.

## Project structure
- tourism_project/data/: source dataset
- tourism_project/model_building/: validation, preparation, and training scripts
- tourism_project/deployment/: Streamlit app, dependencies, and trained model
- tourism_project/reports/: model evaluation and tuning results
- .github/workflows/pipeline.yml: automated machine learning workflow

## Modeling approach
The model uses 13 customer-profile predictors and excludes identifiers,
sales-contact information, pitch details, and follow-up information.
Identical predictor profiles are kept together during splitting and
cross-validation. Hyperparameters are selected using average precision.

## Automation
GitHub Actions validates data, prepares splits, trains and evaluates the
model, preserves MLflow results as workflow artifacts, and commits the
trained model and reports.

## Deployment
Deploy tourism_project/deployment/app.py on Streamlit Community Cloud
using Python 3.11.

## Limitations
The target records historical package purchases. Performance does not
establish future purchase rates for a newly introduced wellness package.
Customer details must be available before outreach to support the
intended prediction use case.
