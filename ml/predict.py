from pathlib import Path
import joblib
import pandas as pd

from ml.config import MODEL_DIR


def load_best_model(model_name: str = "random_forest"):
    """Loads trained model pipeline binary."""
    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run 'python -m ml.train' first."
        )
    return joblib.load(model_path)


def predict_compatibility(sample_data: dict, model_name: str = "random_forest") -> dict:
    """Predicts organ matching probability for a given donor-recipient pair."""
    pipeline = load_best_model(model_name)
    
    df_sample = pd.DataFrame([sample_data])

    prediction = pipeline.predict(df_sample)[0]
    probabilities = pipeline.predict_proba(df_sample)[0]

    match_probability = float(probabilities[1])
    status = "Compatible" if prediction == 1 else "Not Compatible"

    return {
        "status": status,
        "match_probability": round(match_probability * 100, 2),
        "confidence_score": round(float(max(probabilities)) * 100, 2),
        "model_used": model_name,
    }


if __name__ == "__main__":
    # Complete sample donor & recipient input matching raw CSV structure
    sample_pair = {
        "Patient_Age": 45,
        "Patient_Weight": 70.0,
        "Patient_BMI": 24.2,
        "Patient_BloodType": "A",
        "Diagnosis_Result": "CKD Stage 4",
        "Biological_Markers": "HLA-A Match",
        "Donor_Age": 42,
        "Donor_Weight": 68.0,
        "Donor_BloodType": "A",
        "Donor_Medical_Approval": "Yes",
        "RealTime_Organ_HealthScore": 0.92,
        "Organ_Condition_Alert": "Normal",
        "Donor_Min_Age": 18,
        "Donor_Max_Age": 65,
        "Donor_Min_Weight": 50.0,
        "Donor_Max_Weight": 100.0,
    }

    result = predict_compatibility(sample_pair)
    print("\n" + "=" * 45)
    print("      Lifelink-AI Organ Match Result      ")
    print("=" * 45)
    print(f"Match Status      : {result['status']}")
    print(f"Match Probability : {result['match_probability']}%")
    print(f"Confidence Score  : {result['confidence_score']}%")
    print(f"Model Engine      : {result['model_used']}")
    print("=" * 45)