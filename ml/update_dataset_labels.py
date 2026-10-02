import numpy as np
import pandas as pd
from ml.config import RAW_DATA_PATH


def check_abo_compatibility(patient_blood: str, donor_blood: str) -> bool:
    """Standard Medical ABO Blood Group Compatibility Rules."""
    p_blood = str(patient_blood).strip().upper()
    d_blood = str(donor_blood).strip().upper()

    # O is Universal Donor, AB is Universal Recipient
    if d_blood == "O":
        return True
    if p_blood == "AB":
        return True
    if p_blood == d_blood:
        return True
    if p_blood == "A" and d_blood in ["A", "O"]:
        return True
    if p_blood == "B" and d_blood in ["B", "O"]:
        return True

    return False


def calculate_medical_match_status(row) -> str:
    """Applies strict medical criteria to determine kidney donor-recipient compatibility."""
    # 1. Check ABO Blood Group Compatibility
    blood_match = check_abo_compatibility(
        row.get("Patient_BloodType", ""), row.get("Donor_BloodType", "")
    )
    if not blood_match:
        return "No"

    # 2. Donor Medical Fitness Flag
    medical_approval = (
        str(row.get("Donor_Medical_Approval", "")).strip().lower() == "yes"
    )
    if not medical_approval:
        return "No"

    # 3. Organ Condition & Real-Time Health Score Check
    organ_alert = (
        str(row.get("Organ_Condition_Alert", "")).strip().lower()
    )
    organ_health = float(row.get("RealTime_Organ_HealthScore", 0))

    if organ_alert == "critical" or organ_health < 0.65:
        return "No"

    # 4. Age & Weight Differential Tolerance
    age_diff = abs(
        float(row.get("Donor_Age", 0)) - float(row.get("Patient_Age", 0))
    )
    weight_diff = abs(
        float(row.get("Donor_Weight", 0)) - float(row.get("Patient_Weight", 0))
    )

    # Medical rule: Compatible if age gap <= 25 years and weight gap <= 30 kg
    if age_diff <= 25 and weight_diff <= 30:
        return "Yes"

    return "No"


def main():
    if not RAW_DATA_PATH.exists():
        print(f"❌ Dataset not found at '{RAW_DATA_PATH}'.")
        return

    print("🔄 Loading dataset and applying medical compatibility logic...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Generate realistic rule-based labels
    df["Match_Status"] = df.apply(calculate_medical_match_status, axis=1)

    # Save back to CSV
    df.to_csv(RAW_DATA_PATH, index=False)

    print("✅ Dataset successfully updated with realistic medical labels!")
    print("\nUpdated Target Distribution:")
    print(df["Match_Status"].value_counts())


if __name__ == "__main__":
    main()