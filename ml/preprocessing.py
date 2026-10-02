import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.feature_engineering import add_engineered_features

TARGET = "Match_Status"

# Identifiers, post-decision, and non-predictive columns to drop before ML training
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
    """Loads raw CSV dataset and drops duplicate rows."""
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Remove exact duplicate rows if present
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        print(f"🧹 Removed {initial_len - len(df)} duplicate row(s).")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    return df


def prepare_dataset(df: pd.DataFrame):
    """Applies feature engineering, target binary encoding, and builds sklearn preprocessor pipeline."""
    # 1. Apply Feature Engineering (Calculates compatibility scores)
    data = add_engineered_features(df)

    # 2. Encode target column safely (Yes -> 1, No -> 0)
    y = data[TARGET].astype(str).str.lower().str.strip().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Target contains unsupported or missing values.")

    # 3. Drop Target and non-predictive metadata/leakage columns
    X = data.drop(columns=[TARGET])
    columns_to_drop = [column for column in DROP_COLUMNS if column in X.columns]
    X = X.drop(columns=columns_to_drop)

    # 4. Identify column types dynamically (supports Pandas 2 & 3 without warnings)
    categorical_features = X.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # 5. Numerical Pipeline (Median Imputation + Standard Scaling)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 6. Categorical Pipeline (Most Frequent Imputation + One-Hot Encoding)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 7. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return X, y, preprocessor


if __name__ == "__main__":
    from pathlib import Path

    raw_path = Path("data/raw/kidney_dataset.csv")
    if raw_path.exists():
        df_raw = load_dataset(raw_path)
        X, y, preprocessor = prepare_dataset(df_raw)

        # Test preprocessor transformation
        X_trans = preprocessor.fit_transform(X)

        print("✅ Preprocessing Pipeline Executed Successfully!")
        print(f"Features matrix shape: {X.shape}")
        print(f"Transformed matrix shape: {X_trans.shape}")
        print("\nTarget Class Distribution:")
        print(y.value_counts())
    else:
        print(f"❌ Dataset file not found at '{raw_path}'.")