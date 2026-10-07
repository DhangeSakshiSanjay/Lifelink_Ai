# LifeLink-AI Update

## Current update

- Added an explicit hierarchical compatibility gate with per-stage reasons.
- Exposed successful gate stages in the AI matching screen.
- Added AI result metadata describing the model source.
- Added an 8-week implementation plan for project documentation.
- Added a feature-to-code map for viva/project review.
- Preserved the existing ML, SHAP, approval, transport, security, audit and reporting modules.

## Important academic boundary

The application remains a B.Tech academic decision-support prototype. It does not claim clinical validation, live traffic/GPS, or medically validated organ-preservation limits.

## Reliability Update — Notifications, Routing and Model Evaluation
- Fixed Notifications so opening the tab no longer marks all messages as read.
- Added unread count badge, per-notification Mark Read, Mark All Read, and first-run demo notification.
- Added workflow notifications for approvals/rejections and transport route/status changes.
- Upgraded route optimisation with a travel-time weighted A* graph, more route nodes/alternatives, stored route geometry, and Leaflet route drawing.
- Added optional OSRM/OpenStreetMap road routing (`ENABLE_LIVE_ROUTING=1`) with automatic offline A* fallback.
- Added route optimisation simulation output with route alternatives and map preview.
- Fixed model-evaluation leakage by removing heart `survtime_days` (post-outcome variable) from prediction features.
- Added organ-specific model configurations so Random Forest, Gradient Boosting, XGBoost and LightGBM produce genuine, distinguishable evaluation results.
- Added model-comparison regression tests.

## End-to-end allocation workflow upgrade
- Added `MatchRequest` for recipient selection, donor consent, doctor approval, hospital approval and admin final authorization.
- Added donor lifecycle enforcement: `DOCTOR_PENDING -> HOSPITAL_PENDING -> ADMIN_PENDING -> AVAILABLE` with `RESERVED/ALLOCATED/COMPLETED` states.
- Added recipient lifecycle enforcement: doctor approval before `REQUEST_ACTIVE` and AI matching.
- Matching now considers only `AVAILABLE` donors and persists ranked candidates separately from selected requests.
- Added automatic waiting-recipient recheck when a donor becomes `AVAILABLE`.
- Added event-driven notification references, explicit read controls, donor request accept/reject actions and no auto-read on page open.
- Route planning is locked until `FINAL_APPROVED`; transport authorization is a separate state before movement begins.
- Added four 5,000-row synthetic/academic organ datasets (kidney, liver, heart, lung) and a leakage-safe preprocessing + stratified holdout + 5-fold CV training pipeline for RF/GB/XGBoost/LightGBM.
- Added workflow integrity tests and preserved existing AI Lab, SHAP, analytics, security, audit, route and transport modules.
