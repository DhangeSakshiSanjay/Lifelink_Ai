import pandas as pd
import numpy as np


BLOOD_TYPES = ["A", "B", "AB", "O"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Age difference
    data["Age_Difference"] = (
        data["Donor_Age"] - data["Patient_Age"]
    ).abs()

    # Weight difference
    data["Weight_Difference"] = (
        data["Donor_Weight"] - data["Patient_Weight"]
    ).abs()

    # Weight compatibility ratio
    data["Weight_Ratio"] = (
        data["Donor_Weight"] /
        data["Patient_Weight"].replace(0, np.nan)
    )

    # Blood group compatibility
    data["Blood_Group_Compatible"] = (
        data["Patient_BloodType"]
        == data["Donor_BloodType"]
    ).astype(int)

    # Medical approval
    data["Medical_Approval_Flag"] = (
        data["Donor_Medical_Approval"]
        .str.lower()
        .eq("yes")
        .astype(int)
    )

    # Organ health percentage
    data["Organ_Health_Percentage"] = (
        data["RealTime_Organ_HealthScore"] * 100
    )

    # Critical condition flag
    data["Critical_Organ_Flag"] = (
        data["Organ_Condition_Alert"]
        .str.lower()
        .eq("critical")
        .astype(int)
    )

    # Diagnosis severity
    diagnosis_mapping = {
        "CKD Stage 4": 2,
        "CKD Stage 5": 3,
        "ESRD": 4,
    }

    data["Diagnosis_Severity"] = (
        data["Diagnosis_Result"]
        .map(diagnosis_mapping)
        .fillna(0)
    )

    # Age compatibility score
    data["Age_Compatibility_Score"] = np.exp(
        -data["Age_Difference"] / 20
    )

    # Weight compatibility score
    data["Weight_Compatibility_Score"] = np.exp(
        -data["Weight_Difference"] / 20
    )

    return data