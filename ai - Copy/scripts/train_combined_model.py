# scripts/train_combined_model.py
# Train model du doan ca cardio va diabetes tu dataset merged

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

DATA_PATH = Path("data/merged_cardio_diabetes.csv")
MODELS_DIR = Path("data/models")
REPORTS_DIR = Path("data/reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT_RF = MODELS_DIR / "combined_cardio_diabetes_rf_model.pkl"
MODEL_OUT_LR = MODELS_DIR / "combined_cardio_diabetes_lr_model.pkl"
MODEL_OUT_SVM = MODELS_DIR / "combined_cardio_diabetes_svm_model.pkl"
MODEL_OUT_XGB = MODELS_DIR / "combined_cardio_diabetes_xgb_model.pkl"

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def build_pipeline_rf(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    rf = RandomForestClassifier(
        n_estimators=100,  # Reduced to reduce overfitting
        max_depth=10,      # Limit depth
        min_samples_split=10,  # Require more samples
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    # Multi-output cho 2 targets
    multi_rf = MultiOutputClassifier(rf)
    return Pipeline([("pre", pre), ("clf", multi_rf)])

def build_pipeline_lr(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    lr = LogisticRegression(
        C=0.1,  # Stronger regularization
        penalty='l2',
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    # Multi-output cho 2 targets
    multi_lr = MultiOutputClassifier(lr)
    return Pipeline([("pre", pre), ("clf", multi_lr)])

def build_pipeline_svm(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    svm = SVC(
        C=1.0,
        kernel='rbf',
        probability=True,  # For ROC
        random_state=42,
        class_weight="balanced"
    )
    # Multi-output cho 2 targets
    multi_svm = MultiOutputClassifier(svm)
    return Pipeline([("pre", pre), ("clf", multi_svm)])

def build_pipeline_xgb(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    xgb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    # Multi-output cho 2 targets
    multi_xgb = MultiOutputClassifier(xgb)
    return Pipeline([("pre", pre), ("clf", multi_xgb)])

def plot_roc(y_true, y_prob, out_roc, label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"{label} AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {label}")
    plt.legend(); plt.tight_layout(); plt.savefig(out_roc); plt.close()

def evaluate_model(pipe, X_train, X_test, y_train, y_test, model_name):
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    y_prob_train = pipe.predict_proba(X_train)
    y_prob_test = pipe.predict_proba(X_test)
    
    results = {}
    for i, t in enumerate(["cardio", "diabetes"]):
        print(f"\n==== {model_name.upper()} - {t.upper()} ====")
        # Test metrics
        report_test = classification_report(y_test[t], y_pred_test[:, i], output_dict=True)
        auc_test = roc_auc_score(y_test[t], y_prob_test[i][:, 1])
        acc_test = report_test['accuracy']
        print("Test Set:")
        print(classification_report(y_test[t], y_pred_test[:, i]))
        print(f"Test ROC-AUC: {auc_test:.4f}")
        
        # Train metrics
        report_train = classification_report(y_train[t], y_pred_train[:, i], output_dict=True)
        auc_train = roc_auc_score(y_train[t], y_prob_train[i][:, 1])
        acc_train = report_train['accuracy']
        print("Train Set:")
        print(classification_report(y_train[t], y_pred_train[:, i]))
        print(f"Train ROC-AUC: {auc_train:.4f}")
        
        # Check for overfitting
        acc_diff = acc_train - acc_test
        auc_diff = auc_train - auc_test
        print(f"Accuracy Diff (Train - Test): {acc_diff:.4f}")
        print(f"AUC Diff (Train - Test): {auc_diff:.4f}")
        if acc_diff > 0.05 or auc_diff > 0.05:
            print("⚠️  Possible Overfitting Detected!")
        else:
            print("✅ No significant overfitting.")
        
        plot_roc(y_test[t], y_prob_test[i][:, 1], REPORTS_DIR / f"{model_name.lower().replace(' ', '_')}_{t}_roc_curve.png", f"{model_name} - {t}")
        results[t] = {"report_test": report_test, "auc_test": auc_test, "acc_test": acc_test, "report_train": report_train, "auc_train": auc_train, "acc_train": acc_train}
    return results

def main():
    df = load_data(DATA_PATH)

    # Tach targets
    targets = ["cardio", "diabetes"]
    df_targets = df[targets].copy()
    X = df.drop(columns=targets)

    # Phan loai cot
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    cat_cols = ["gender", "hypertension", "heart_disease", "cholesterol", "smoking_history"]

    print("\n===== Dataset Overview =====")
    print("Rows:", len(df))
    print("Numeric cols:", num_cols)
    print("Categorical cols:", cat_cols)
    for t in targets:
        if t in df_targets.columns:
            print(f"{t} class balance:")
            print(df_targets[t].dropna().value_counts().to_string())
    print("============================\n")

    # Drop rows where both targets are NaN
    mask = df_targets.notna().any(axis=1)
    X = X[mask]
    df_targets = df_targets[mask]

    # Fill NaN targets with 0 (assume healthy if not known)
    df_targets = df_targets.fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, df_targets, test_size=0.2, random_state=42
    )

    # Random Forest
    print("Training Random Forest...")
    rf_results = evaluate_model(build_pipeline_rf(num_cols, cat_cols), X_train, X_test, y_train, y_test, "Random Forest")
    joblib.dump(build_pipeline_rf(num_cols, cat_cols).fit(X_train, y_train), MODEL_OUT_RF)
    print("Saved RF model:", MODEL_OUT_RF)

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_results = evaluate_model(build_pipeline_lr(num_cols, cat_cols), X_train, X_test, y_train, y_test, "Logistic Regression")
    joblib.dump(build_pipeline_lr(num_cols, cat_cols).fit(X_train, y_train), MODEL_OUT_LR)
    print("Saved LR model:", MODEL_OUT_LR)

    # SVM (skip due to slow training)
    print("\nSkipping SVM due to slow training on large dataset...")

    # XGBoost
    print("\nTraining XGBoost...")
    xgb_results = evaluate_model(build_pipeline_xgb(num_cols, cat_cols), X_train, X_test, y_train, y_test, "XGBoost")
    joblib.dump(build_pipeline_xgb(num_cols, cat_cols).fit(X_train, y_train), MODEL_OUT_XGB)
    print("Saved XGB model:", MODEL_OUT_XGB)

    # Comparison
    print("\n===== COMPARISON =====")
    models = ["Random Forest", "Logistic Regression", "XGBoost"]
    results_dict = {"Random Forest": rf_results, "Logistic Regression": lr_results, "XGBoost": xgb_results}
    for t in targets:
        print(f"\n{t.upper()}:")
        for model in models:
            auc_val = results_dict[model][t]["auc_test"]
            print(f"  {model}: AUC = {auc_val:.4f}")
        best_model = max(models, key=lambda m: results_dict[m][t]["auc_test"])
        print(f"  -> Best: {best_model}")

    # Print table for report
    print("\n===== TABLE FOR REPORT =====")
    print("| Thuật toán | Cardio AUC | Diabetes AUC |")
    print("|------------|------------|---------------|")
    for model in models:
        cardio_auc = results_dict[model]["cardio"]["auc_test"]
        diabetes_auc = results_dict[model]["diabetes"]["auc_test"]
        print(f"| {model} | {cardio_auc:.4f} | {diabetes_auc:.4f} |")
    print("=============================\n")

if __name__ == "__main__":
    main()