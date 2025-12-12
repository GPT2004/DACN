# api/app_heart_form.py
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# === Đường dẫn model ===
BUNDLE_PATH = Path("data/models/heart_rf_model.pkl")

app = Flask(__name__, template_folder="templates")

# === Cột raw chuẩn của heart.csv (UCI) ===
HEART_RAW_COLS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]

# === Load model/bundle ===
_loaded = joblib.load(BUNDLE_PATH)
if isinstance(_loaded, dict):
    # hỗ trợ bundle {"model": pipeline_or_estimator, "feature_columns": [...]}
    model = _loaded.get("model", _loaded)
    feature_cols = _loaded.get("feature_columns", HEART_RAW_COLS)
else:
    model = _loaded
    feature_cols = HEART_RAW_COLS

# === Cấu hình hiển thị và phân loại mức nguy cơ ===
DEFAULT_THRESHOLD = 0.5
BANDS = [
    (0.0,  0.20, "Thấp",        "bg-green-100 text-green-800"),
    (0.20, 0.50, "Trung bình",  "bg-yellow-100 text-yellow-800"),
    (0.50, 1.01, "Cao",         "bg-red-100 text-red-800"),
]
def risk_band(p: float):
    for lo, hi, name, css in BANDS:
        if lo <= p < hi:
            return name, css
    return "N/A", ""

# === Trang form ===
@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None

    # các field đúng theo UCI Heart
    fields = {
        "Age": {"type": "number", "label": "Tuổi", "step": "1"},
        "Sex": {"type": "select", "label": "Giới tính", "options": ["M", "F"]},
        "ChestPainType": {"type": "select", "label": "Kiểu đau ngực", "options": ["ATA", "NAP", "ASY", "TA"]},
        "RestingBP": {"type": "number", "label": "Huyết áp nghỉ (mmHg)", "step": "1"},
        "Cholesterol": {"type": "number", "label": "Cholesterol (mg/dL)", "step": "1"},
        "FastingBS": {"type": "select", "label": "Đường huyết lúc đói (>120mg/dL?)", "options": [0, 1]},
        "RestingECG": {"type": "select", "label": "Điện tâm đồ", "options": ["Normal", "ST", "LVH"]},
        "MaxHR": {"type": "number", "label": "Nhịp tim tối đa", "step": "1"},
        "ExerciseAngina": {"type": "select", "label": "Đau ngực khi gắng sức", "options": ["Y", "N"]},
        "Oldpeak": {"type": "number", "label": "Oldpeak (ST depression)", "step": "0.1"},
        "ST_Slope": {"type": "select", "label": "Độ dốc ST", "options": ["Up", "Flat", "Down"]},
    }

    if request.method == "POST":
        try:
            # threshold (tùy chọn)
            try:
                threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))
                threshold = max(0.0, min(1.0, threshold))
            except Exception:
                threshold = DEFAULT_THRESHOLD

            # lấy input
            data = {}
            for k, meta in fields.items():
                v = request.form.get(k, "").strip()
                if meta["type"] == "number":
                    data[k] = float(v) if v != "" else None
                else:
                    # giữ string/category đúng như khi train (pipeline xử lý)
                    data[k] = v if v != "" else None

            # chuẩn hoá DataFrame theo feature_cols
            X = pd.DataFrame([data])
            for c in feature_cols:
                if c not in X.columns:
                    X[c] = None
            X = X[feature_cols]

            # dự đoán
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

    return render_template(
        "heart_form.html",  # tạo file template này (có thể copy từ diabetes_form.html và sửa fields)
        fields=fields,
        result=result,
        error=error,
        default_threshold=DEFAULT_THRESHOLD
    )

# === API JSON ===
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
    app.run(host="0.0.0.0", port=8001, debug=True)
