from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


class SHAPExplainer:
    """
    Explainability service for the Lifelink-AI
    donor-recipient compatibility model.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.pipeline = joblib.load(self.model_path)

        self.preprocessor = (
            self.pipeline.named_steps["preprocessor"]
        )

        self.model = (
            self.pipeline.named_steps["model"]
        )

    def _transform_input(
        self,
        input_data: pd.DataFrame
    ):
        return self.preprocessor.transform(input_data)

    def _get_feature_names(self) -> list[str]:
        """
        Get feature names after preprocessing.
        """

        feature_names = (
            self.preprocessor
            .get_feature_names_out()
        )

        return list(feature_names)

    def explain(
        self,
        input_data: pd.DataFrame
    ) -> dict[str, Any]:

        transformed_data = self._transform_input(
            input_data
        )

        feature_names = self._get_feature_names()

        # Tree-based models such as
        # Random Forest, XGBoost, LightGBM,
        # and Gradient Boosting can be explained
        # using TreeExplainer.
        explainer = shap.TreeExplainer(
            self.model
        )

        shap_values = explainer.shap_values(
            transformed_data
        )

        # Binary classification handling
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        feature_contributions = []

        for name, value in zip(
            feature_names,
            values
        ):
            feature_contributions.append(
                {
                    "feature": name,
                    "contribution": float(value),
                    "direction": (
                        "positive"
                        if value > 0
                        else "negative"
                    ),
                }
            )

        feature_contributions.sort(
            key=lambda item:
            abs(item["contribution"]),
            reverse=True
        )

        return {
            "model": type(
                self.model
            ).__name__,

            "top_positive_factors": [
                item
                for item in feature_contributions
                if item["contribution"] > 0
            ][:10],

            "top_negative_factors": [
                item
                for item in feature_contributions
                if item["contribution"] < 0
            ][:10],

            "all_features": feature_contributions,
        }