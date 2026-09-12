from typing import Any

import pandas as pd


class DonorRanker:

    def __init__(self, model):
        self.model = model

    def rank(
        self,
        donor_recipient_pairs: pd.DataFrame,
        identifiers: list[dict[str, Any]],
    ) -> pd.DataFrame:

        probabilities = (
            self.model.predict_proba(
                donor_recipient_pairs
            )[:, 1]
        )

        results = pd.DataFrame(
            identifiers
        )

        results[
            "compatibility_score"
        ] = probabilities * 100

        results = results.sort_values(
            by="compatibility_score",
            ascending=False
        ).reset_index(
            drop=True
        )

        results["rank"] = (
            results.index + 1
        )

        return results