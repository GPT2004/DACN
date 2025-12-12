# api/app_diabetes_form.py
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

BUNDLE_PATH = Path("data/models/diabetes_risk_model.pkl")

app = Flask(__name__, template_folder="templates")

# ===== Load model bundle (đã train theo dataset 100k) =====
bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
feature_cols = bundle["feature_columns"]

# Cấu hình hiển thị và phân loại mức nguy cơ
DEFAULT_THRESHOLD = 0.5   # ngưỡng ra label 0/1
BANDS = [
    (0.0, 0.20, "Thấp",   "bg-green-100 text-green-800"),
    (0.20, 0.50, "Trung bình", "bg-yellow-100 text-yellow-800"),
    (0.50, 1.01, "Cao",    "bg-red-100 text-red-800"),
]

def risk_band(p: float):
    for lo, hi, name, css in BANDS:
        if lo <= p < hi:
            return name, css
    return "N/A", ""

# ===== Trang form =====
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    # các field đúng theo dataset 100k
    fields = {
        "gender": {"type": "select", "label": "Giới tính", "options": ["male", "female", "other"]},
        "age": {"type": "number", "label": "Tuổi", "step": "1"},
        "hypertension": {"type": "select", "label": "Tăng huyết áp", "options": [0, 1]},
        "heart_disease": {"type": "select", "label": "Bệnh tim mạch", "options": [0, 1]},
        "smoking_history": {"type": "select", "label": "Hút thuốc", "options": ["never", "former", "current", "not current", "ever", "unknown"]},
        "bmi": {"type": "number", "label": "BMI", "step": "0.1"},
        "HbA1c_level": {"type": "number", "label": "HbA1c (%)", "step": "0.1"},
        "blood_glucose_level": {"type": "number", "label": "Đường huyết (mg/dL)", "step": "1"},
    }

    if request.method == "POST":
        try:
            # Lấy ngưỡng từ form (tuỳ chọn)
            try:
                threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))
                threshold = max(0.0, min(1.0, threshold))
            except Exception:
                threshold = DEFAULT_THRESHOLD

            # Lấy input
            data = {}
            for k in fields.keys():
                v = request.form.get(k, "").strip()
                if k in ("age", "bmi", "HbA1c_level", "blood_glucose_level"):
                    data[k] = float(v) if v != "" else None
                elif k in ("hypertension", "heart_disease"):
                    # nhận 0/1
                    data[k] = int(v) if v != "" else None
                else:
                    data[k] = v if v != "" else None

            # Chuẩn hoá DataFrame đúng thứ tự cột khi train
            X = pd.DataFrame([data])
            for c in feature_cols:
                if c not in X.columns:
                    X[c] = None
            X = X[feature_cols]

            # Dự đoán
            prob = float(model.predict_proba(X)[:, 1][0])
            pred = int(prob >= threshold)
            band_name, band_css = risk_band(prob)

            result = {
                "input": data,
                "prob": round(prob, 4),
                "pred": pred,
                "threshold": threshold,
                "band": band_name,
                "band_css": band_css,
            }

        except Exception as e:
            error = f"Lỗi xử lý đầu vào: {e}"

    return render_template("diabetes_form.html",
                           fields=fields,
                           result=result,
                           error=error,
                           default_threshold=DEFAULT_THRESHOLD)

# ===== API JSON (tuỳ chọn) =====
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
        X = pd.DataFrame([payload])
        for c in feature_cols:
            if c not in X.columns:
                X[c] = None
        X = X[feature_cols]
        prob = float(model.predict_proba(X)[:, 1][0])
        pred = int(prob >= DEFAULT_THRESHOLD)
        return jsonify({"ok": True, "pred_prob": prob, "pred_label": pred, "threshold": DEFAULT_THRESHOLD})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    # Flask dev server
    app.run(host="0.0.0.0", port=8000, debug=True)
