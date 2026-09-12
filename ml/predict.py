from pathlib import Path

import joblib
import pandas as pd


class MatchingModel:

    def __init__(
        self,
        model_path: str | Path
    ):

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        self.pipeline = joblib.load(
            self.model_path
        )

    def predict(
        self,
        data: pd.DataFrame
    ) -> dict:

        probability = (
            self.pipeline
            .predict_proba(data)[0][1]
        )

        prediction = int(
            probability >= 0.5
        )

        return {
            "prediction": prediction,
            "compatibility_score":
                round(
                    probability * 100,
                    2
                ),
            "recommendation":
                (
                    "HIGH_COMPATIBILITY"
                    if probability >= 0.75
                    else
                    "REVIEW_REQUIRED"
                    if probability >= 0.50
                    else
                    "LOW_COMPATIBILITY"
                ),
        }