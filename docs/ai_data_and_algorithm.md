# AI Data and Algorithm Overview

## Data
- Domains: symptoms, diagnoses, lab orders, prescriptions, vitals.
- Sources: CSV/JSON from `ai/data`, DB tables for patients/records/prescriptions.
- Key files: `ai/data/vi2en_symptoms.json` (symptom mapping), `ai/data/department_labels_vi.json` (dept labels), trained models under `ai/data/models/`.
- Features (examples): age, gender, BMI, blood pressure, glucose, cholesterol, symptoms (multi-hot), lab results.

## Data Flow
1) User inputs (forms) -> API payload.
2) API normalizes/validates -> model-ready vector.
3) Model inference (XGBoost) -> probabilities / classes.
4) Post-process: thresholding, top-N, label mapping (vi/en), rounding.
5) Response -> frontend renders (percent, badges, risk text).

## XGBoost: How It Works
- Ensemble of gradient-boosted decision trees.
- Each tree is a weak learner; trees are added sequentially to correct residual errors.
- Objective: minimize loss (e.g., logistic for classification) with regularization (gamma, lambda) to control overfit.
- Splits chosen by gain; learning_rate shrinks each tree’s contribution; max_depth/child_weight balance bias-variance.
- Output: sum of tree scores passed through logistic -> probability.

## Models in This Project
- Symptom model: multi-class classifier mapping symptoms -> department/condition; multi-label encoded; top-k probabilities.
- Cardio model: XGBoost binary/ordinal risk on vitals (BP, cholesterol, BMI, etc.).
- Diabetes model: XGBoost binary risk on glucose, BMI, BP, heredity, age.
- Combined model: aggregates features from multiple domains for general triage.

## Two AI Services
- Service 1 (port 5001): Symptom/disease suggestion.
  - Input: demographic + symptom list.
  - Output: top diagnoses/departments with probabilities; optional advice.
- Service 2 (port 5000): Risk scoring (cardio/diabetes/combined).
  - Input: vitals + labs + demographic.
  - Output: risk probabilities and textual explanation.
- Both expose HTTP endpoints; CORS enabled for frontend.

## Inference Steps (both services)
1) Receive JSON -> validate -> encode features.
2) Load XGBoost model from disk.
3) Run `model.predict_proba` -> probabilities.
4) Map class indices to labels -> sort -> take top-N.
5) Build response with percentages, labels, and optional hints.

## Thresholds & Post-Processing
- Probabilities often clipped to 0.01–0.99, displayed as 1–99%.
- Top-3 or top-5 used for UI; low-confidence results still shown with lower opacity.
- Missing fields default to safe neutral values to avoid crashes.

## Limitations & Notes
- Models depend on training distribution; out-of-distribution inputs reduce reliability.
- Not a diagnostic device; outputs are decision support only.
- Keep mappings (vi/en) in sync with model label order.
- Monitor drift: re-train if input statistics shift.
