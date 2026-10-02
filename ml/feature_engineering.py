import numpy as np
import pandas as pd

BLOOD_TYPES = ["A", "B", "AB", "O"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates biological, compatibility, and severity features for the Lifelink-AI dataset."""
    data = df.copy()

    # 1. Age difference & exponential compatibility score
    if "Donor_Age" in data.columns and "Patient_Age" in data.columns:
        data["Age_Difference"] = (
            data["Donor_Age"] - data["Patient_Age"]
        ).abs()
        data["Age_Compatibility_Score"] = np.exp(
            -data["Age_Difference"] / 20.0
        )

    # 2. Weight difference & body size ratio
    if "Donor_Weight" in data.columns and "Patient_Weight" in data.columns:
        data["Weight_Difference"] = (
            data["Donor_Weight"] - data["Patient_Weight"]
        ).abs()
        data["Weight_Ratio"] = data["Donor_Weight"] / data[
            "Patient_Weight"
        ].replace(0, np.nan)
        data["Weight_Compatibility_Score"] = np.exp(
            -data["Weight_Difference"] / 20.0
        )

    # 3. ABO Blood Group exact compatibility flag
    if "Patient_BloodType" in data.columns and "Donor_BloodType" in data.columns:
        data["Blood_Group_Compatible"] = (
            data["Patient_BloodType"] == data["Donor_BloodType"]
        ).astype(int)

    # 4. Donor Medical Fitness Flag
    if "Donor_Medical_Approval" in data.columns:
        data["Medical_Approval_Flag"] = (
            data["Donor_Medical_Approval"]
            .astype(str)
            .str.lower()
            .eq("yes")
            .astype(int)
        )

    # 5. Scale Organ Health Score to Percentage (0 to 100)
    if "RealTime_Organ_HealthScore" in data.columns:
        data["Organ_Health_Percentage"] = (
            data["RealTime_Organ_HealthScore"] * 100
        )

    # 6. Critical Alert Flag
    if "Organ_Condition_Alert" in data.columns:
        data["Critical_Organ_Flag"] = (
            data["Organ_Condition_Alert"]
            .astype(str)
            .str.lower()
            .eq("critical")
            .astype(int)
        )

    # 7. Numerical mapping for Medical Condition Severity
    if "Diagnosis_Result" in data.columns:
        diagnosis_mapping = {
            "CKD Stage 4": 2,
            "CKD Stage 5": 3,
            "ESRD": 4,
        }
        data["Diagnosis_Severity"] = (
            data["Diagnosis_Result"].map(diagnosis_mapping).fillna(0)
        )

    return data


# Allow testing directly from terminal
if __name__ == "__main__":
    import sys
    from pathlib import Path

    raw_path = Path("data/raw/kidney_dataset.csv")
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
        df_featured = add_engineered_features(df_raw)
        print("✅ Feature Engineering Executed Successfully!")
        print(f"Original shape: {df_raw.shape}")
        print(f"New shape after features: {df_featured.shape}")
        print("\nNewly added columns:")
        new_cols = [c for c in df_featured.columns if c not in df_raw.columns]
        print(new_cols)
    else:
        print(
            f"❌ Dataset not found at '{raw_path}'. Place kidney_dataset.csv in data/raw/"
        )