import joblib
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from ml.config import MODEL_DIR, RAW_DATA_PATH
from ml.preprocessing import load_dataset, prepare_dataset


def train_balanced_models():
    """
    Trains ML models using class-weight balancing and SMOTE sampling 
    to eliminate model bias caused by dataset imbalance.
    """
    print("📂 Loading dataset...")
    df = load_dataset(RAW_DATA_PATH)

    print("⚙️ Preprocessing features and applying transformers...")
    X, y, preprocessor = prepare_dataset(df)

    # Calculate class imbalance ratio for XGBoost (866 / 134 ≈ 6.46)
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    scale_pos_weight_value = num_neg / max(num_pos, 1)

    print(f"📊 Class Distribution -> Incompatible (0): {num_neg} | Compatible (1): {num_pos}")
    print(f"⚖️ Applied Minority Class Weighting Factor: {scale_pos_weight_value:.2f}")

    # Train-Test Split (Stratified to maintain identical ratio in test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models configured explicitly for IMBALANCED DATASETS
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",  # Automatically adjusts weights inversely proportional to class frequencies
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight_value,  # Heavily penalizes missing positive compatible pairs
            eval_metric="logloss",
            random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight_value,  # Handles class imbalance natively
            random_state=42,
            verbose=-1,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print("\n🚀 Training Models with Imbalance Corrections...")

    for name, model in models.items():
        print(f"\n⚡ Training model: {name}...")

        # Build full executable pipeline (Preprocessing + Model)
        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        full_pipeline.fit(X_train, y_train)

        # Evaluate performance on unseen test data
        y_pred = full_pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "pipeline": full_pipeline,
        }

        # Save pipeline binary
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(full_pipeline, model_path)

    # Summary Benchmark Table
    print("\n" + "=" * 65)
    print("        Lifelink-AI Balanced Model Benchmark Summary        ")
    print("=" * 65)
    print(f"{'Model Name':<20} | {'F1-Score':<8} | {'Recall':<8} | {'Precision':<9} | {'Accuracy':<8}")
    print("-" * 65)

    best_model_name = None
    best_f1 = -1.0

    for name, metrics in results.items():
        print(
            f"{name:<20} | {metrics['f1']:.4f}   | {metrics['recall']:.4f}   | "
            f"{metrics['precision']:.4f}    | {metrics['accuracy']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name

    print("=" * 65)
    print(f"🏆 Top Performing Unbiased Model (F1-Score): {best_model_name}\n")


if __name__ == "__main__":
    train_balanced_models()