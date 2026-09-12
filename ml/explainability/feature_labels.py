FEATURE_LABELS = {

    "Age_Difference":
        "Age difference between donor and recipient",

    "Weight_Difference":
        "Weight difference",

    "Weight_Ratio":
        "Donor-recipient weight ratio",

    "Blood_Group_Compatible":
        "Blood group compatibility",

    "Medical_Approval_Flag":
        "Donor medical approval",

    "Organ_Health_Percentage":
        "Organ health",

    "Critical_Organ_Flag":
        "Critical organ condition",

    "Diagnosis_Severity":
        "Recipient diagnosis severity",

    "Age_Compatibility_Score":
        "Age compatibility",

    "Weight_Compatibility_Score":
        "Weight compatibility",
}


def get_readable_feature_name(
    feature_name: str
) -> str:

    clean_name = feature_name

    if "__" in clean_name:
        clean_name = clean_name.split(
            "__",
            1
        )[1]

    for key, value in FEATURE_LABELS.items():

        if clean_name.startswith(key):
            return value

    return clean_name.replace(
        "_",
        " "
    ).title()