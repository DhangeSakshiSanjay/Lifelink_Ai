import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.feature_engineering import add_engineered_features


TARGET = "Match_Status"


DROP_COLUMNS = [
    "Patient_ID",
    "Donor_ID",
    "Organ_Tracking_ID",
    "Timestamp_Organ_Scanned",
    "Organ_Status",
    "Predicted_Survival_Chance",
    "Organ_Required",
    "Organ_Donated",
]


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    return df


def prepare_dataset(df: pd.DataFrame):
    data = add_engineered_features(df)

    y = (
        data[TARGET]
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if y.isna().any():
        raise ValueError(
            "Target contains unsupported values."
        )

    X = data.drop(columns=[TARGET])

    columns_to_drop = [
        column
        for column in DROP_COLUMNS
        if column in X.columns
    ]

    X = X.drop(columns=columns_to_drop)

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return X, y, preprocessor