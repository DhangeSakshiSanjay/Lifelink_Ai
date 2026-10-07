# LifeLink-AI V2
## An AI-Driven Smart Organ Matching and Transportation Decision Support System

LifeLink-AI is a final-year B.Tech academic prototype that combines **multi-factor AI-assisted donor-recipient ranking**, **explainable AI (SHAP)**, **priority analysis**, **hospital capability checks**, and **travel-time route optimization** into one auditable case-management workflow.

> **Important:** This is a decision-support/academic system. It is not a clinical allocation engine, does not establish medical suitability, and does not claim live GPS/traffic or clinically validated organ-preservation limits.

## Major modules

1. Role-based authentication and authorization
2. Donor and recipient registry
3. Multi-factor compatibility gate
4. Existing ML ranking pipeline (Random Forest / Gradient Boosting / XGBoost / LightGBM when available)
5. Organ-specific models already present in the supplied project
6. SHAP feature explanations with deterministic fallback
7. Match confidence and operational risk
8. Recipient priority score
9. Model comparison and 5-fold cross-validation research lab
10. Dataset quality validation and experimental fairness audit
11. What-if analysis
12. Doctor approval gate
13. Hospital capability and approval gate
14. Government/Admin verification gate
15. Case timeline and audit trail
16. A* travel-time route optimization
17. Traffic scenario simulation and route re-optimization
18. Ground / emergency ground / air-ambulance academic transport mode comparison
19. Transport feasibility and remaining-time calculation
20. Leaflet + OpenStreetMap route visualization
21. Transport status history
22. Notifications
23. Before/after operational reports
24. Analytics dashboards
25. Security events, account suspension and audit logs

## End-to-end workflow

```text
DONOR: Register -> Doctor -> Hospital -> Admin -> AVAILABLE
RECIPIENT: Register -> Doctor -> ACTIVE REQUEST
AI: AVAILABLE donors -> compatibility -> ML -> priority -> SHAP -> ranking
NO DONOR: ACTIVE/WAITING -> automatic re-check when a new donor becomes AVAILABLE
SELECTED DONOR: Recipient selects -> Donor consent -> Doctor -> Hospital -> Admin
FINAL: FINAL_APPROVED -> Route -> Transport Authorization -> Started -> In Transit -> Arrived -> Completed
```

A separate `MatchRequest` entity records recipient consent, donor consent, doctor approval, hospital approval and admin authorization. No transport route can be calculated before `FINAL_APPROVED`.

## Route optimization

A* uses estimated **travel time** as its edge cost. Therefore the shortest distance is not automatically the selected route.

Traffic factors are configurable:

- LOW = 1.00
- MEDIUM = 1.25
- HIGH = 1.50
- VERY_HIGH = 1.80

The Transport Simulation Lab can compare scenarios and transport modes without claiming live traffic.

## AI methodology

The system keeps the supplied ML pipeline and adds a transparent multi-objective ranking layer:

```text
Hard eligibility
    -> ML compatibility probability
    -> HLA compatibility
    -> blood compatibility
    -> medical-condition indicator
    -> urgency
    -> waiting-time priority
    -> age compatibility
    -> transport feasibility
    -> final score + confidence + risk
```

The feature weights are configurable in `app/ai_lab.py` and are academic parameters, not medical rules.

## Research/evaluation

Open **AI Research Lab** after login to view:

- 5-fold cross-validation
- accuracy / precision / recall / F1 / ROC-AUC
- F1 standard deviation
- dataset quality
- class balance
- outlier counts
- experimental fairness audit

## What-if analysis

For a case, the system can change configurable factors such as:

- HLA compatibility
- antibody indicator
- urgency
- waiting days

and recompute the decision-support score so the user can see which factors change the ranking.

## Demo accounts

| Role | Email | Password |
|---|---|---|
| Admin / Government Security Admin | `admin@lifelink.ai` | `admin123` |
| Doctor | `doctor@lifelink.ai` | `doctor123` |
| Hospital | `hospital@lifelink.ai` | `hospital123` |
| Donor | `donor@lifelink.ai` | `donor123` |
| Recipient | `recipient@lifelink.ai` | `recipient123` |

A first-run demonstration case is seeded in `DOCTOR_PENDING` state so the complete workflow can be demonstrated without auto-approving any medical decision.

## Installation on Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open:

`http://127.0.0.1:5000`

## Environment variables

Copy `.env.example` to `.env` and change `SECRET_KEY` for any real deployment.

`ROUTE_MODE=static` keeps the academic road graph as the deterministic fallback.

## Project structure

```text
LifeLink-AI/
├── app/
│   ├── ml/
│   │   ├── engine.py
│   │   ├── organ_models.py
│   │   └── models/
│   ├── ai_lab.py
│   ├── models.py
│   ├── routes.py
│   ├── transport.py
│   ├── config.py
│   ├── templates/
│   └── static/
├── datasets/
├── tests_route_optimizer.py
├── requirements.txt
└── run.py
```

## Academic limitations

- The supplied kidney benchmark and general matching benchmark include synthetic/academic data.
- No clinical validation is claimed.
- Hospital capability and transport-mode factors are configurable demonstration parameters.
- Static A* graph is the reliable offline fallback; live traffic/GPS requires an external provider and is not claimed by default.
- Route time and organ time windows are configurable academic parameters, not medical guidance.
- Fairness metrics are experimental diagnostics, not evidence of clinical fairness.


## Workflow and ML reliability updates
- Donors registered through the portal start in `DOCTOR_PENDING` and cannot enter matching until hospital and admin authorization also pass.
- Recipient requests start in `DOCTOR_PENDING`; only doctor-approved requests become active.
- Ranked candidates are persisted separately from the selected donor request. A donor rejection reopens the recipient request.
- When a donor reaches `AVAILABLE`, active waiting recipient requests are automatically rechecked and compatible recipients are notified.
- Notifications include read/unread state and workflow references; opening the page does not mark them read.
- Four synthetic/academic datasets (kidney, liver, heart, lung) are generated without row duplication or target leakage. Training uses preprocessing, stratification, 5-fold CV and real held-out metrics.
- The four model families remain Random Forest, Gradient Boosting, XGBoost and LightGBM.

## Recent reliability fixes
- Notifications no longer auto-mark as read; unread badges and Mark All Read are supported.
- Transport routing uses travel-time weighted A* offline and can optionally use OSRM/OpenStreetMap road geometry with `ENABLE_LIVE_ROUTING=1`; it automatically falls back to A* if online routing is unavailable.
- Model evaluation removes the heart dataset post-outcome `survtime_days` leakage and uses a stratified 75/25 holdout with model-specific regularisation, so model metrics are not artificially identical.
