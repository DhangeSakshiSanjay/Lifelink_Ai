from typing import Any

import pandas as pd


class CompatibilityEngine:
    """
    Rule-based eligibility layer.

    This layer does NOT replace the ML model.
    It removes clearly ineligible candidates before
    AI ranking.
    """

    def check(
        self,
        recipient: dict[str, Any],
        donor: dict[str, Any],
    ) -> dict[str, Any]:

        reasons = []

        eligible = True

        # Donor medical approval
        if str(
            donor.get(
                "Donor_Medical_Approval",
                ""
            )
        ).lower() != "yes":

            eligible = False

            reasons.append(
                "Donor medical approval is not available."
            )

        # Organ type
        recipient_organ = str(
            recipient.get(
                "Organ_Required",
                "Kidney"
            )
        ).lower()

        donor_organ = str(
            donor.get(
                "Organ_Donated",
                "Kidney"
            )
        ).lower()

        if recipient_organ != donor_organ:

            eligible = False

            reasons.append(
                "Required organ and donated organ "
                "do not match."
            )

        return {
            "eligible": eligible,
            "reasons": reasons,
        }