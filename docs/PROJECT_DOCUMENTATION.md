# Project Documentation — Rising Water

## Abstract
Flooding can cause major disruption to communities and infrastructure. This project demonstrates a machine-learning workflow for estimating a flood-risk category from environmental and infrastructure indicators. The dataset contains 20 predictor fields and a continuous `FloodProbability` value. For classification, the project derives a binary label using a documented threshold of 0.50. Decision Tree, Random Forest, K-Nearest Neighbors and XGBoost classifiers are trained and compared. The selected model is saved and integrated into a Flask web application.

## Problem statement
Build an educational prototype that accepts a consistent set of environmental and infrastructure scores and returns a model-generated risk category. The prototype demonstrates how data science models can be integrated into a simple web application.

## Objectives
1. Inspect and prepare the supplied flood dataset.
2. Derive a binary classification target from the continuous probability field.
3. Train four classification algorithms using a reproducible train/test split.
4. Compare models with multiple evaluation metrics.
5. Save the selected model and connect it to a Flask interface.
6. Validate user input and clearly communicate model limitations.

## Dataset description
`dataset/flood.csv` contains 50,000 rows. The 20 input fields are scores from 0 to 10. `FloodProbability` is the continuous target supplied with the dataset. The classification target is defined as `FloodProbability >= 0.50`. The threshold is a modelling choice for this academic exercise, not an official flood-risk threshold.

## System architecture
```text
User
  |
  v
Flask Web Form (20 indicators)
  |
  v
Server-side validation (numeric, 0 to 10)
  |
  v
Pandas feature row (fixed feature order)
  |
  v
Saved XGBoost classifier
  |
  v
Class + high-class model probability
  |
  v
Result page + safety disclaimer
```

## Methodology
1. Load the CSV file using pandas.
2. Remove duplicate and incomplete rows.
3. Separate 20 predictors from `FloodProbability`.
4. Convert the continuous target into binary labels using 0.50.
5. Split the data into 80% training and 20% testing with stratification.
6. Fit a `StandardScaler` for KNN; tree-based models use original score values.
7. Train Decision Tree, Random Forest, KNN and XGBoost.
8. Calculate accuracy, precision, recall, F1-score and ROC-AUC.
9. Select the model with the highest F1-score and save the model.
10. Load the saved XGBoost model in Flask and validate submitted values.

## Results
See `reports/model_metrics.json` and `reports/classification_report.txt` for the measured benchmark. On the included data and split, XGBoost achieved 93.15% accuracy and 93.29% F1-score. The result is not a real-world forecasting validation.

## Three usage scenarios
### Scenario 1 — Educational early-warning demonstration
A student enters sample indicator scores for a hypothetical flood-prone area. The application returns a higher- or lower-risk class to demonstrate the workflow. It must not be treated as an actual evacuation advisory.

### Scenario 2 — Comparing regions in a classroom exercise
A user enters different hypothetical score combinations for separate regions and compares the resulting model outputs. The scores must come from a consistent source and should not be interpreted as live monitoring.

### Scenario 3 — Model validation
A student or analyst reviews the held-out test metrics and classification report to understand model performance, limitations and the effect of the chosen target threshold.

## Limitations and future scope
- The dataset is tabular and its score fields are not direct live weather-station measurements.
- The binary label is derived from a threshold on the supplied target.
- No geographic, time-series, river-gauge or live weather feed is integrated.
- Test-set metrics do not guarantee generalization to new regions or future events.
- Future work could use verified time-stamped hydrological data, location-aware validation, probability calibration, external validation and expert review.

## Conclusion
Rising Water demonstrates an end-to-end student ML workflow: data preparation, model comparison, evaluation, model persistence and Flask deployment. It is a learning prototype only and must not replace official flood forecasting or emergency services.
