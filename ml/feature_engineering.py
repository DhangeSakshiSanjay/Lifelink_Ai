import numpy as np
import pandas as pd


def check_blood_compatibility(donor_blood: str, recipient_blood: str) -> bool:
    """
    Checks ABO blood group compatibility for organ transplantation.
    
    Standard Medical ABO Matching Matrix:
    - Donor O  -> Recipients: O, A, B, AB (Universal Donor)
    - Donor A  -> Recipients: A, AB
    - Donor B  -> Recipients: B, AB
    - Donor AB -> Recipients: AB Only
    """
    if pd.isna(donor_blood) or pd.isna(recipient_blood):
        return False

    donor_blood = str(donor_blood).strip().upper()
    recipient_blood = str(recipient_blood).strip().upper()

    compatibility_map = {
        "O": ["O", "A", "B", "AB"],
        "A": ["A", "AB"],
        "B": ["B", "AB"],
        "AB": ["AB"],
    }

    allowed_recipients = compatibility_map.get(donor_blood, [])
    return recipient_blood in allowed_recipients


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes biological disparity, physical ratio, clinical severity,
    and ABO compatibility features across the donor-recipient pair.
    """
    data = df.copy()

    # 1. Physical & Demographics Disparity Metrics
    if "Patient_Age" in data.columns and "Donor_Age" in data.columns:
        data["Age_Difference"] = (data["Patient_Age"] - data["Donor_Age"]).abs()
        data["Age_Compatibility_Score"] = np.where(data["Age_Difference"] <= 15, 1.0, 0.5)

    if "Patient_Weight" in data.columns and "Donor_Weight" in data.columns:
        data["Weight_Difference"] = (data["Patient_Weight"] - data["Donor_Weight"]).abs()
        data["Weight_Ratio"] = data["Patient_Weight"] / (data["Donor_Weight"] + 1e-5)
        # Optimal organ size ratio is between 0.8 and 1.25
        data["Weight_Compatibility_Score"] = np.where(
            (data["Weight_Ratio"] >= 0.8) & (data["Weight_Ratio"] <= 1.25), 1.0, 0.5
        )

    # 2. Hard ABO Blood Group Compatibility Flag
    if "Donor_BloodType" in data.columns and "Patient_BloodType" in data.columns:
        data["Blood_Group_Compatible"] = data.apply(
            lambda row: check_blood_compatibility(
                row["Donor_BloodType"], row["Patient_BloodType"]
            ),
            axis=1,
        )

    # 3. Medical Clearance & Health Flags
    if "Donor_Medical_Approval" in data.columns:
        data["Medical_Approval_Flag"] = data["Donor_Medical_Approval"].apply(
            lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true", "approved"] else 0
        )

    if "RealTime_Organ_HealthScore" in data.columns:
        data["Organ_Health_Percentage"] = data["RealTime_Organ_HealthScore"] * 100.0
        data["Critical_Organ_Flag"] = np.where(data["RealTime_Organ_HealthScore"] < 0.6, 1, 0)

    # 4. Clinical Diagnosis Severity Mapping (Kidney Urgency)
    if "Diagnosis_Result" in data.columns:
        severity_map = {
            "CKD Stage 5": 3,
            "End Stage Renal Disease": 3,
            "ESRD": 3,
            "CKD Stage 4": 2,
            "CKD Stage 3": 1,
        }
        data["Diagnosis_Severity"] = (
            data["Diagnosis_Result"].map(severity_map).fillna(1).astype(int)
        )

    return data


if __name__ == "__main__":
    sample_df = pd.DataFrame([
        {
            "Patient_Age": 45,
            "Donor_Age": 42,
            "Patient_Weight": 70.0,
            "Donor_Weight": 68.0,
            "Donor_BloodType": "A",
            "Patient_BloodType": "A",
            "Donor_Medical_Approval": "Yes",
            "RealTime_Organ_HealthScore": 0.92,
            "Diagnosis_Result": "CKD Stage 4",
        }
    ])
    featured = add_engineered_features(sample_df)
    print("✅ Feature Engineering Verification Passed!")
    print(f"Total Columns ({len(featured.columns)}): {list(featured.columns)}")