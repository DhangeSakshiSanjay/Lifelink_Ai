import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml.config import ARTIFACT_DIR, RAW_DATA_PATH
from ml.predict import load_best_model
from ml.preprocessing import prepare_dataset


def evaluate_all_models():
    """Evaluates all trained models on the test split and saves metrics + confusion matrices."""
    # 1. Load dataset & prepare features
    df = pd.read_csv(RAW_DATA_PATH)
    X, y, preprocessor = prepare_dataset(df)

    # 2. Replicate train/test split (same random_state as train.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = ["random_forest", "xgboost", "logistic_regression"]
    results = {}

    print("\n" + "=" * 60)
    print("      Lifelink-AI Model Evaluation & Quality Metrics      ")
    print("=" * 60)

    for model_name in models:
        try:
            pipeline = load_best_model(model_name)
        except FileNotFoundError:
            continue

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = round(accuracy_score(y_test, y_pred) * 100, 2)
        prec = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
        rec = round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
        roc = round(roc_auc_score(y_test, y_prob) * 100, 2)

        results[model_name] = {
            "Accuracy": f"{acc}%",
            "Precision": f"{prec}%",
            "Recall": f"{rec}%",
            "F1_Score": f"{f1}%",
            "ROC_AUC": f"{roc}%",
        }

        print(f"\n📊 Model: {model_name.upper()}")
        print(f"   • Accuracy  : {acc}%")
        print(f"   • Precision : {prec}% (Low false positives)")
        print(f"   • Recall    : {rec}% (Low false negatives)")
        print(f"   • F1-Score  : {f1}% (Balanced metric)")
        print(f"   • ROC-AUC   : {roc}%")

        # Save Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Incompatible", "Compatible"],
            yticklabels=["Incompatible", "Compatible"],
        )
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        cm_path = ARTIFACT_DIR / f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

    # Save metrics JSON artifact
    metrics_path = ARTIFACT_DIR / "model_evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print(f"✅ Metric reports saved to '{metrics_path}'")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_all_models()