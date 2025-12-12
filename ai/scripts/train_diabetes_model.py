# scripts/train_diabetes_rf_model.py
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
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)

DATA_PATH = Path("tieuduong/diabetes_prediction_dataset.csv")
MODELS_DIR = Path("data/models")
REPORTS_DIR = Path("data/reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = MODELS_DIR / "diabetes_rf_model.pkl"

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_target(df: pd.DataFrame) -> str:
    for c in ["diabetes", "Outcome", "outcome", "label", "target"]:
        if c in df.columns:
            return c
    raise ValueError("Không tìm thấy cột mục tiêu (diabetes/Outcome/outcome/label/target)")

def normalize_y(y: pd.Series) -> pd.Series:
    if y.dtype.kind in "biu":
        return y.astype(int)
    tmp = y.astype(str).str.strip().str.lower().map({
        "1": 1, "yes": 1, "true": 1, "positive": 1,
        "0": 0, "no": 0, "false": 0, "negative": 0
    })
    if tmp.isna().any():
        codes, _ = pd.factorize(y)
        tmp = codes
    return pd.Series(tmp, index=y.index).astype(int)

def build_pipeline(num_cols, cat_cols):
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
        ("cat", cat_pipe, cat_cols)
    ])
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    return Pipeline([("pre", pre), ("clf", rf)])

def plot_curve(y_true, y_prob, out_roc, out_pr):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Random Forest")
    plt.legend(); plt.tight_layout(); plt.savefig(out_roc); plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve – Random Forest")
    plt.legend(); plt.tight_layout(); plt.savefig(out_pr); plt.close()

def plot_confusion(y_true, y_pred, out_png):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix – Random Forest")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def main():
    df = load_data(DATA_PATH)
    target_col = find_target(df)

    y = normalize_y(df[target_col])
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    print("\n===== Dataset Overview =====")
    print("Rows:", len(df))
    print("Numeric cols:", len(num_cols), "| Categorical cols:", len(cat_cols))
    print("Class balance (y):")
    print(y.value_counts().rename(index={0:"class 0",1:"class 1"}).to_string())
    print("============================\n")

    pipe = build_pipeline(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]

    print("🔍 Thông tin đánh giá mô hình")
    print(classification_report(y_test, y_pred, digits=3))
    auc_val = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc_val:.4f}")

    plot_curve(y_test, y_prob,
               REPORTS_DIR / "diabetes_rf_roc_curve.png",
               REPORTS_DIR / "diabetes_rf_pr_curve.png")
    plot_confusion(y_test, y_pred,
                   REPORTS_DIR / "diabetes_rf_confusion_matrix.png")

    print("📊 Saved plots: ROC, PR, Confusion Matrix → data/reports/")
    joblib.dump(pipe, MODEL_OUT)
    print("✅ Saved model:", MODEL_OUT)

if __name__ == "__main__":
    main()
