# LifeLink-AI — 8-Week Implementation Plan

This plan describes the development sequence for the current academic prototype. Weeks 2–4 are directly supported by the submitted weekly assessment reports; Weeks 5–8 describe the implementation sequence used to complete the integrated prototype.

| Week | Task | Main implementation |
|---|---|---|
| 1 | Problem and requirement analysis | Problem definition, users, requirements and end-to-end workflow |
| 2 | System design and database | `app/models.py`, `app/routes.py`, `app/templates/` |
| 3 | Dataset and feature preparation | `datasets/kidney/`, `app/ml/organ_models.py`, `app/ml/engine.py` |
| 4 | ML model training and comparison | Random Forest, Gradient Boosting, XGBoost, LightGBM; evaluation metrics |
| 5 | AI matching and donor ranking | `app/ai_lab.py`, `app/ml/engine.py`, `matching.html` |
| 6 | Human approval workflow | Doctor → Hospital → Admin gates in `app/routes.py` |
| 7 | Transport and route optimization | `app/transport.py`, `transport.html`, A* travel-time routing |
| 8 | Integration, testing and documentation | End-to-end testing, audit trail, reports, analytics and final documentation |

## Final workflow

Registration → compatibility gate → ML prediction → multi-factor ranking → SHAP explanation → doctor approval → hospital approval → admin verification → route optimization → transport feasibility → transport lifecycle → reports.

## Academic limitation

The system is a decision-support prototype. Its synthetic/benchmark datasets, compatibility rules, transport parameters and risk scores are not clinical validation or medical advice.
