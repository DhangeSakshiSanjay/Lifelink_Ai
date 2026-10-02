from pathlib import Path
import joblib
import pandas as pd

from ml.config import MODEL_DIR
from ml.feature_engineering import add_engineered_features, check_blood_compatibility

# Columns to drop before model prediction
DROP_COLUMNS = [
    "Patient_ID",
    "Donor_ID",
    "Organ_Tracking_ID",
    "Timestamp_Organ_Scanned",
    "Organ_Status",
    "Predicted_Survival_Chance",
    "Organ_Required",
    "Organ_Donated",
    "Match_Status",
]

# Explicit list of text/categorical features
EXPLICIT_CATEGORICAL_COLS = [
    "Patient_BloodType",
    "Diagnosis_Result",
    "Biological_Markers",
    "Donor_BloodType",
    "Donor_Medical_Approval",
    "Organ_Condition_Alert",
]


def load_best_model(model_name: str = "gradient_boosting"):
    """
    Loads trained model pipeline binary. 
    Defaults to 'gradient_boosting' (Top Benchmark Model - F1: 0.9818, Recall: 1.0000).
    """
    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run 'python -m ml.train' first."
        )
    return joblib.load(model_path)


def predict_compatibility(sample_data: dict, model_name: str = "gradient_boosting") -> dict:
    """
    Predicts organ matching probability for a given donor-recipient pair.
    Applies strict clinical safety rules (ABO compatibility & donor medical approval)
    to guarantee 0% probability on medically unsafe donor-recipient pairs.
    """
    # --- STEP 1: HARD CLINICAL SAFETY GUARDRAILS ---
    donor_blood = sample_data.get("Donor_BloodType", "")
    patient_blood = sample_data.get("Patient_BloodType", "")
    medical_approval = str(sample_data.get("Donor_Medical_Approval", "")).strip().lower()

    # Rule A: ABO Blood Group Compatibility
    is_blood_ok = check_blood_compatibility(donor_blood, patient_blood)

    # Rule B: Donor Medical Clearance
    is_approved = medical_approval in ["yes", "1", "true", "approved"]

    # Safety Override: Force 0% match if hard clinical constraints are violated
    if not is_blood_ok or not is_approved:
        rejection_reasons = []
        if not is_blood_ok:
            rejection_reasons.append(
                f"ABO Blood Group Incompatibility (Donor {donor_blood} cannot donate to Recipient {patient_blood})"
            )
        if not is_approved:
            rejection_reasons.append("Donor Medical Clearance Rejected")

        return {
            "status": "Incompatible",
            "match_probability": 0.0,
            "confidence_score": 100.0,
            "model_used": model_name,
            "clinical_warning": f"Hard Rejection: {'; '.join(rejection_reasons)}",
        }

    # --- STEP 2: ML MODEL EVALUATION (For Medically Viable Pairs) ---
    pipeline = load_best_model(model_name)

    df_sample = pd.DataFrame([sample_data])
    df_featured = add_engineered_features(df_sample)

    cols_to_drop = [col for col in DROP_COLUMNS if col in df_featured.columns]
    df_featured = df_featured.drop(columns=cols_to_drop)

    # Enforce data types to align with trained preprocessor
    for col in df_featured.columns:
        if col in EXPLICIT_CATEGORICAL_COLS:
            df_featured[col] = df_featured[col].astype(str)
        elif pd.api.types.is_bool_dtype(df_featured[col]):
            df_featured[col] = df_featured[col].astype(int)
        else:
            df_featured[col] = pd.to_numeric(df_featured[col], errors="coerce")

    prediction = pipeline.predict(df_featured)[0]
    probabilities = pipeline.predict_proba(df_featured)[0]

    match_probability = float(probabilities[1])
    status = "Compatible" if (prediction == 1 and match_probability >= 0.5) else "Incompatible"

    return {
        "status": status,
        "match_probability": round(match_probability * 100, 2),
        "confidence_score": round(float(max(probabilities)) * 100, 2),
        "model_used": model_name,
        "clinical_warning": "Passed Clinical Safety Guardrails",
    }


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("      Lifelink-AI Clinical Match Evaluation Test      ")
    print("=" * 55)

    # Test Case 1: Valid Compatible Pair (Donor Type A -> Patient Type A)
    valid_pair = {
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

    res1 = predict_compatibility(valid_pair)
    print("\n--- TEST 1: Valid Compatible Pair ---")
    print(f"Match Status      : {res1['status']}")
    print(f"Match Probability : {res1['match_probability']}%")
    print(f"Confidence Score  : {res1['confidence_score']}%")
    print(f"Model Engine      : {res1['model_used']}")
    print(f"Clinical Status   : {res1['clinical_warning']}")

    # Test Case 2: Incompatible Blood Type Pair (Donor Type A -> Patient Type B)
    invalid_blood_pair = valid_pair.copy()
    invalid_blood_pair["Patient_BloodType"] = "B"

    res2 = predict_compatibility(invalid_blood_pair)
    print("\n--- TEST 2: ABO Incompatible Pair ---")
    print(f"Match Status      : {res2['status']}")
    print(f"Match Probability : {res2['match_probability']}%")
    print(f"Confidence Score  : {res2['confidence_score']}%")
    print(f"Model Engine      : {res2['model_used']}")
    print(f"Clinical Warning  : {res2['clinical_warning']}")
    print("=" * 55)