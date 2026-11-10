"""Train and evaluate four flood-risk classifiers.

Run from the repository root:
    python train_model.py
The target is derived from the supplied continuous FloodProbability column:
    1 = FloodProbability >= 0.50, 0 = otherwise.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "flood.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
TARGET = "FloodProbability"
THRESHOLD = 0.50
RANDOM_STATE = 42

def main():
    data = pd.read_csv(DATA)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain '{TARGET}'.")
    data = data.drop_duplicates().dropna()
    features = [column for column in data.columns if column != TARGET]
    X = data[features]
    y = (data[TARGET] >= THRESHOLD).astype("int8")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "Decision Tree": (DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE), X_train, X_test),
        "Random Forest": (RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE), X_train, X_test),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1), X_train_scaled, X_test_scaled),
        "XGBoost": (XGBClassifier(n_estimators=220, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, reg_lambda=1, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE), X_train, X_test),
    }

    results = {}
    fitted = {}
    for name, (estimator, train_data, test_data) in candidates.items():
        estimator.fit(train_data, y_train)
        predicted = estimator.predict(test_data)
        probability = estimator.predict_proba(test_data)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "recall": float(recall_score(y_test, predicted, zero_division=0)),
            "f1": float(f1_score(y_test, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
        }
        fitted[name] = estimator
        print(f"{name}: accuracy={results[name]['accuracy']:.4f}, F1={results[name]['f1']:.4f}")

    winner = max(results, key=lambda name: results[name]["f1"])
    best_model = fitted[winner]
    # Keep the saved artifact format compatible with the selected estimator.
    if winner == "XGBoost":
        best_model.save_model(str(MODEL_DIR / "xgboost_flood_classifier.json"))
        model_file = "xgboost_flood_classifier.json"
        model_format = "xgboost_json"
    else:
        joblib.dump(best_model, MODEL_DIR / "best_flood_classifier.joblib")
        model_file = "best_flood_classifier.joblib"
        model_format = "joblib"

    joblib.dump(scaler, MODEL_DIR / "standard_scaler.joblib")
    (MODEL_DIR / "selected_model.json").write_text(
        json.dumps({"name": winner, "file": model_file, "format": model_format}, indent=2),
        encoding="utf-8",
    )
    prediction = best_model.predict(X_test)
    report = {
        "target_definition": "Flood = 1 when FloodProbability >= 0.50; otherwise 0",
        "threshold": THRESHOLD,
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "test_samples": int(len(y_test)),
        "class_distribution": {"low": int((y_test == 0).sum()), "high": int((y_test == 1).sum())},
        "selected_model": winner,
        "models": results,
    }
    (REPORT_DIR / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, prediction, target_names=["Low risk", "High risk"]), encoding="utf-8"
    )
    (REPORT_DIR / "confusion_matrix.json").write_text(
        json.dumps(confusion_matrix(y_test, prediction).tolist(), indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    print(f"Selected model: {winner}")
    print(f"Reports saved in {REPORT_DIR}")

if __name__ == "__main__":
    main()
