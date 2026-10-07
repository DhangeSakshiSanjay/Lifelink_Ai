# LifeLink-AI — Feature-to-Code Map

## AI/ML
- `app/ml/engine.py`: baseline feature generation, four ML classifiers, train/test evaluation, prediction and SHAP explanation.
- `app/ml/organ_models.py`: organ-specific dataset loading, preprocessing, model training and metric persistence.
- `app/ai_lab.py`: multi-factor ranking, cross-validation, data-quality checks, fairness diagnostics and what-if analysis.

## Matching
- `app/ml/engine.py::compatibility_gate`: hierarchical hard-gate checks.
- `app/ai_lab.py::enhanced_match`: ML score + HLA + ABO + medical indicator + urgency + waiting priority + age + transport feasibility.
- `app/routes.py::matching` and `create_match`: ranked candidates and persistent match records.

## Human-in-the-loop governance
- `app/routes.py::approve`, `reject`, `verify_case`, `reject_case_admin`.
- `app/models.py::Match`: workflow state, decisions, reasons and timestamps.

## Transport
- `app/transport.py`: A* travel-time route search, alternatives, traffic scenarios, ETA, transport modes, feasibility and operational-risk heuristic.
- `app/routes.py::route`, `reoptimize_transport`, `transport_status`.
- `app/models.py::Transport` and `TransportEvent`: route plan and lifecycle history.

## Responsible engineering
- SHAP explanations.
- Dataset quality diagnostics.
- Experimental fairness audit.
- CSRF token protection.
- Role-based permissions and verification.
- Security events and audit logs.
- Before/after reports and case timeline.

All AI, fairness and transport outputs are explicitly academic decision-support outputs.
