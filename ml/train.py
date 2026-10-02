import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from ml.config import (
    ARTIFACT_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEST_SIZE,
)
from ml.preprocessing import load_dataset, prepare_dataset


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(RAW_DATA_PATH)
    X, y, preprocessor = prepare_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
    }

    results = {}

    for name, estimator in models.items():
        print(f"\n⚡ Training model: {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        # Save individual model binary
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(pipeline, model_path)

    # Save metrics JSON artifact
    results_path = ARTIFACT_DIR / "model_comparison.json"
    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)

    best_model = max(results, key=lambda k: results[k]["f1_score"])

    print("\n" + "=" * 55)
    print("        Lifelink-AI Model Benchmark Summary        ")
    print("=" * 55)
    for name, metrics in results.items():
        print(
            f"{name:20} | F1: {metrics['f1_score']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | Accuracy: {metrics['accuracy']:.4f}"
        )
    print("=" * 55)
    print(f"🏆 Top Performing Model (F1-Score): {best_model}")


if __name__ == "__main__":
    main()