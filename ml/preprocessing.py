import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.feature_engineering import add_engineered_features

TARGET = "Match_Status"

# Non-predictive metadata & post-decision columns to drop before ML training
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
    """Loads CSV dataset, removes duplicates, and performs initial health checks."""
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
    """Applies feature engineering, target mapping, and constructs scikit-learn preprocessing pipeline."""
    # 1. Run Feature Engineering
    data = add_engineered_features(df)

    # 2. Map Target Column safely (Yes -> 1, No -> 0)
    y = data[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Target column contains invalid or unmapped values.")

    # 3. Drop Target & Identifier/Leakage Columns
    X = data.drop(columns=[TARGET])
    cols_to_drop = [col for col in DROP_COLUMNS if col in X.columns]
    X = X.drop(columns=cols_to_drop)

    # 4. Safely separate Numeric vs Categorical features
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numerical_features]

    # 5. Robust Pipeline for Numerical Features
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 6. Robust Pipeline for Categorical Features
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

        X_trans = preprocessor.fit_transform(X)

        print("✅ Preprocessing Pipeline Built & Tested Successfully!")
        print(f"Feature matrix (X) shape: {X.shape}")
        print(f"Transformed matrix shape: {X_trans.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
    else:
        print(f"❌ Dataset not found at '{raw_path}'.")