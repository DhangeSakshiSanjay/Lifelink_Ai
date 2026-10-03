import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.feature_engineering import add_engineered_features

TARGET = "Match_Status"

# Non-predictive metadata & leakage columns to drop
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

# Categorical text features requiring One-Hot Encoding
EXPLICIT_CATEGORICAL_COLS = [
    "Patient_BloodType",
    "Diagnosis_Result",
    "Biological_Markers",
    "Donor_BloodType",
    "Donor_Medical_Approval",
    "Organ_Condition_Alert",
]


def load_dataset(path: str) -> pd.DataFrame:
    """Loads CSV dataset, removes duplicates, and validates schema."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    initial_count = len(df)
    df = df.drop_duplicates()
    if initial_count != len(df):
        print(f"🧹 Removed {initial_count - len(df)} duplicate row(s).")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing from dataset.")
    return df


def prepare_dataset(df: pd.DataFrame):
    """
    Applies feature engineering, extracts target variable,
    and constructs a scikit-learn ColumnTransformer.
    """
    # 1. Generate engineered features
    data = add_engineered_features(df)

    # 2. Extract target column (Yes -> 1, No -> 0)
    if TARGET in data.columns:
        y = data[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        if y.isna().any():
            raise ValueError("Target column contains unmapped or null values.")
    else:
        y = None

    # 3. Drop metadata columns
    cols_to_drop = [col for col in DROP_COLUMNS if col in data.columns]
    X = data.drop(columns=cols_to_drop)

    # 4. Separate Numeric vs Categorical feature sets
    categorical_features = [col for col in EXPLICIT_CATEGORICAL_COLS if col in X.columns]
    for col in categorical_features:
        X[col] = X[col].astype(str)

    numerical_features = [col for col in X.columns if col not in categorical_features]
    for col in numerical_features:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # 5. Transformation Pipelines
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 6. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return X, y, preprocessor


if __name__ == "__main__":
    from pathlib import Path
    from ml.config import RAW_DATA_PATH

    if Path(RAW_DATA_PATH).exists():
        df_raw = load_dataset(RAW_DATA_PATH)
        X, y, preprocessor = prepare_dataset(df_raw)
        X_trans = preprocessor.fit_transform(X)

        print("\n✅ Preprocessing Pipeline Built Successfully!")
        print(f"Feature matrix (X) shape : {X.shape}")
        print(f"Transformed matrix shape : {X_trans.shape}")
        print(f"Target distribution:\n{y.value_counts()}")