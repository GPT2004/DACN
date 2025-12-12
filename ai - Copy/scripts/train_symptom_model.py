from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# === utils (VI -> EN + cấu hình) ===
from symptom_utils import vi_to_en_symptom_map, GENERAL_DEPT, THRESHOLD

DATA_MERGED = Path("data/symptom_checker_merged.csv")
DATA_FALLBACK = Path("data/symptom_checker_from_kaggle.csv")
MODELS_DIR = Path("data/models")
REPORTS_DIR = Path("data/reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    if DATA_MERGED.exists():
        print(f"Using {DATA_MERGED}")
        return pd.read_csv(DATA_MERGED, encoding="utf-8-sig")
    else:
        print(f"Using {DATA_FALLBACK}")
        return pd.read_csv(DATA_FALLBACK, encoding="utf-8-sig")

def build_pipeline_rf():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000)),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

def build_pipeline_lr():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000)),
        ("clf", LogisticRegression(
            C=0.1,
            penalty='l2',
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

def build_pipeline_xgb():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000)),
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ])

def evaluate_model(pipe, X_train, X_test, y_train, y_test, model_name):
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print(f"\n==== {model_name.upper()} ====")
    print("Train Accuracy:", acc_train)
    print("Test Accuracy:", acc_test)
    
    acc_diff = acc_train - acc_test
    print(f"Accuracy Diff (Train - Test): {acc_diff:.4f}")
    if acc_diff > 0.05:
        print("⚠️  Possible Overfitting Detected!")
    else:
        print("✅ No significant overfitting.")
    
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred_test))
    
    return {"acc_train": acc_train, "acc_test": acc_test}

def main():
    data = load_data()

    # Thống kê trước khi chỉnh ENT
    before_counts = data["department"].astype(str).value_counts()
    print("\n===== Dataset Statistics (before) =====")
    print(before_counts)
    print("======================================\n")

    # Upsample ENT if needed (from original script)
    # Assuming upsample is done or skip for simplicity
    data_aug = data  # Skip upsample for now

    # Chuẩn hoá text
    X_raw = data_aug["text"].astype(str)
    y = data_aug["department"].astype(str)
    X = [vi_to_en_symptom_map(x) for x in X_raw]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Random Forest
    print("Training Random Forest...")
    rf_results = evaluate_model(build_pipeline_rf(), X_train, X_test, y_train, y_test, "Random Forest")
    joblib.dump(build_pipeline_rf().fit(X_train, y_train), MODELS_DIR / "symptom_rf_model.pkl")
    print("Saved RF model")

    # Logistic Regression
    print("Training Logistic Regression...")
    lr_results = evaluate_model(build_pipeline_lr(), X_train, X_test, y_train, y_test, "Logistic Regression")
    joblib.dump(build_pipeline_lr().fit(X_train, y_train), MODELS_DIR / "symptom_lr_model.pkl")
    print("Saved LR model")

    # XGBoost
    print("\nTraining XGBoost...")
    xgb_results = evaluate_model(build_pipeline_xgb(), X_train, X_test, y_train, y_test, "XGBoost")
    joblib.dump(build_pipeline_xgb().fit(X_train, y_train), MODELS_DIR / "symptom_xgb_model.pkl")
    print("Saved XGB model")

    # Comparison
    print("\n===== COMPARISON =====")
    models = ["Random Forest", "Logistic Regression", "XGBoost"]
    results_dict = {"Random Forest": rf_results, "Logistic Regression": lr_results, "XGBoost": xgb_results}
    for model in models:
        acc_val = results_dict[model]["acc_test"]
        print(f"  {model}: Accuracy = {acc_val:.4f}")
    best_model = max(models, key=lambda m: results_dict[m]["acc_test"])
    print(f"  -> Best: {best_model}")

    # Print table for report
    print("\n===== TABLE FOR REPORT =====")
    print("| Thuật toán | Accuracy (Test) |")
    print("|------------|-----------------|")
    for model in models:
        acc_val = results_dict[model]["acc_test"]
        print(f"| {model} | {acc_val:.4f} |")
    print("=============================\n")

    # Check overfitting for best
    best_results = results_dict[best_model]
    acc_diff = best_results["acc_train"] - best_results["acc_test"]
    print(f"\nBest model {best_model} overfitting check: Diff = {acc_diff:.4f}")
    if acc_diff > 0.05:
        print("⚠️  Best model may have overfitting.")
    else:
        print("✅ Best model has no significant overfitting.")

if __name__ == "__main__":
    main()
