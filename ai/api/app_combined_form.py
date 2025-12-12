# api/app_combined_form.py
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re

HEART_BUNDLE_PATH = Path(__file__).parent / "../data/models/heart_rf_model.pkl"
DIAB_BUNDLE_PATH = Path(__file__).parent / "../data/models/diabetes_risk_model.pkl"

app = Flask(__name__, template_folder="../templates")
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:8080"]}})

# load heart model bundle
_hb = joblib.load(HEART_BUNDLE_PATH)
if isinstance(_hb, dict):
    heart_model = _hb.get("model", _hb)
    heart_features = _hb.get("feature_columns", None)
else:
    heart_model = _hb
    heart_features = None

# load diabetes model bundle
_db = joblib.load(DIAB_BUNDLE_PATH)
if isinstance(_db, dict):
    diab_model = _db.get("model", _db)
    diab_features = _db.get("feature_columns", None)
else:
    diab_model = _db
    diab_features = None

DEFAULT_THRESHOLD = 0.5
BANDS = [
    (0.0, 0.20, "Thấp", "bg-green-100 text-green-800"),
    (0.20, 0.50, "Trung bình", "bg-yellow-100 text-yellow-800"),
    (0.50, 1.01, "Cao", "bg-red-100 text-red-800"),
]

def risk_band(p: float):
    for lo, hi, name, css in BANDS:
        if lo <= p < hi:
            return name, css
    return "N/A", ""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted = None

    # Heart form fields (match `app_heart_form.py` keys)
    heart_fields = {
        "Age": {"type": "number", "label": "Tuổi", "step": "1", "desc": "Tuổi của bạn (năm)"},
        "Sex": {"type": "select", "label": "Giới tính", "options": ["M", "F"], "desc": "M/F"},
        "ChestPainType": {"type": "select", "label": "Kiểu đau ngực", "options": ["ATA", "NAP", "ASY", "TA"], "desc": "Kiểu đau ngực"},
        "RestingBP": {"type": "number", "label": "Huyết áp nghỉ (mmHg)", "step": "1", "desc": "Huyết áp"},
        "Cholesterol": {"type": "number", "label": "Cholesterol (1/2/3)", "step": "1", "desc": "1: Bình thường, 2: Cao, 3: Rất cao"},
        "FastingBS": {"type": "select", "label": "FastingBS (>120 mg/dL)", "options": [0, 1], "desc": "0: No, 1: Yes"},
        "RestingECG": {"type": "select", "label": "Điện tâm đồ", "options": ["Normal", "ST", "LVH"], "desc": "ECG"},
        "MaxHR": {"type": "number", "label": "Nhịp tim tối đa", "step": "1", "desc": "MaxHR"},
        "ExerciseAngina": {"type": "select", "label": "Đau ngực khi gắng sức", "options": ["Y", "N"], "desc": "Y/N"},
        "Oldpeak": {"type": "number", "label": "Oldpeak (ST depression)", "step": "0.1", "desc": "Oldpeak"},
        "ST_Slope": {"type": "select", "label": "Độ dốc ST", "options": ["Up", "Flat", "Down"], "desc": "ST Slope"},
    }

    # Diabetes form fields (match `app_diabetes_form.py` keys)
    diabetes_fields = {
        "gender": {"type": "select", "label": "Giới tính", "options": ["male", "female", "other"], "desc": "Giới tính"},
        "age": {"type": "number", "label": "Tuổi", "step": "1", "desc": "Tuổi"},
        "hypertension": {"type": "select", "label": "Tăng huyết áp", "options": [0, 1], "desc": "0/1"},
        "heart_disease": {"type": "select", "label": "Bệnh tim mạch", "options": [0, 1], "desc": "0/1"},
        "smoking_history": {"type": "select", "label": "Hút thuốc", "options": ["never", "former", "current", "not current", "ever", "unknown"], "desc": "Lịch sử hút thuốc"},
        "bmi": {"type": "number", "label": "BMI", "step": "0.01", "desc": "Chỉ số khối cơ thể (kg/m²)"},
        "HbA1c_level": {"type": "number", "label": "HbA1c (%)", "step": "0.1", "desc": "Mức HbA1c"},
        "blood_glucose_level": {"type": "number", "label": "Đường huyết (mg/dL)", "step": "1", "desc": "Đường huyết"},
    }

    # Explanations for select option values (displayed under each select)
    heart_option_help = {
        "ChestPainType": {
            "ATA": "Typical angina — đau thắt ngực điển hình (do thiếu máu cơ tim)",
            "NAP": "Non-anginal pain — đau không do thiếu máu cơ tim",
            "ASY": "Asymptomatic — không triệu chứng (có tổn thương nhưng không đau)",
            "TA": "Atypical angina — đau ngực không điển hình",
        },
        "Cholesterol": {
            "1": "Bình thường",
            "2": "Cao",
            "3": "Rất cao (mức nguy cơ cao hơn)",
        },
        "FastingBS": {
            "0": "Không (không cao hơn 120 mg/dL)",
            "1": "Có (trên 120 mg/dL)",
        },
        "RestingECG": {
            "Normal": "Bình thường",
            "ST": "Thay đổi đoạn ST/T",
            "LVH": "Tăng khối thất trái (Left ventricular hypertrophy)",
        },
        "ExerciseAngina": {
            "Y": "Có đau ngực khi gắng sức",
            "N": "Không có đau ngực khi gắng sức",
        },
        "ST_Slope": {
            "Up": "Dốc lên (thường là tốt hơn)",
            "Flat": "Phẳng",
            "Down": "Dốc xuống (cần chú ý)",
        },
        "Sex": {
            "M": "Nam",
            "F": "Nữ",
        }
    }

    diabetes_option_help = {
        "smoking_history": {
            "never": "Chưa từng hút thuốc",
            "former": "Đã từng hút, hiện đã bỏ",
            "current": "Đang hút",
            "not current": "Không còn hút hiện tại",
            "ever": "Từng hút (không rõ trạng thái)",
            "unknown": "Không rõ",
        },
        "gender": {
            "male": "Nam",
            "female": "Nữ",
            "other": "Khác / không muốn tiết lộ",
        }
    }

    # combine helpers so template can look up by key
    option_help = {**heart_option_help, **diabetes_option_help}

    if request.method == "POST":
        try:
            # raw inputs from forms
            heart_raw = {k: request.form.get(k, "").strip() for k in heart_fields.keys()}
            diab_raw = {k: request.form.get(k, "").strip() for k in diabetes_fields.keys()}

            def to_num(x, dtype=float):
                if x is None or x == "":
                    return None
                try:
                    return dtype(x)
                except Exception:
                    return None

            # Map inputs into feature names expected by combined model
            mapped = {
                "age": to_num(diab_raw.get("age") or heart_raw.get("Age"), float),
                "gender": ( (diab_raw.get("gender") or "").lower() or ("male" if heart_raw.get("Sex") == "M" else ("female" if heart_raw.get("Sex") == "F" else "")) ),
                "hypertension": to_num(diab_raw.get("hypertension"), int),
                "heart_disease": to_num(diab_raw.get("heart_disease"), int),
                "cholesterol": to_num(heart_raw.get("Cholesterol"), int),
                "blood_glucose_level": to_num(diab_raw.get("blood_glucose_level"), float),
                "bmi": to_num(diab_raw.get("bmi"), float),
                "HbA1c_level": to_num(diab_raw.get("HbA1c_level"), float),
                "smoking_history": diab_raw.get("smoking_history") or None,
            }

            # normalize gender to lowercase or None
            if mapped.get("gender"):
                try:
                    mapped["gender"] = str(mapped["gender"]).lower()
                except Exception:
                    mapped["gender"] = None

            # convert None -> np.nan so imputers recognize missing
            for k, v in mapped.items():
                if v is None:
                    mapped[k] = np.nan

            # Build model-specific DataFrames and coerce types
            # Heart prediction: build DataFrame for heart_model
            heart_df = pd.DataFrame([{
                'Age': to_num(heart_raw.get('Age'), float),
                'Sex': heart_raw.get('Sex') or None,
                'ChestPainType': heart_raw.get('ChestPainType') or None,
                'RestingBP': to_num(heart_raw.get('RestingBP'), float),
                'Cholesterol': to_num(heart_raw.get('Cholesterol'), float),
                'FastingBS': to_num(heart_raw.get('FastingBS'), float),
                'RestingECG': heart_raw.get('RestingECG') or None,
                'MaxHR': to_num(heart_raw.get('MaxHR'), float),
                'ExerciseAngina': heart_raw.get('ExerciseAngina') or None,
                'Oldpeak': to_num(heart_raw.get('Oldpeak'), float),
                'ST_Slope': heart_raw.get('ST_Slope') or None,
            }])
            if heart_features:
                for c in heart_features:
                    if c not in heart_df.columns:
                        heart_df[c] = None
                heart_df = heart_df[heart_features]
            for col in ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']:
                if col in heart_df.columns:
                    heart_df[col] = pd.to_numeric(heart_df[col], errors='coerce')

            # Diabetes prediction: build DataFrame for diab_model
            diab_df = pd.DataFrame([{
                'gender': (diab_raw.get('gender') or '').lower() or ( 'male' if heart_raw.get('Sex') == 'M' else ('female' if heart_raw.get('Sex') == 'F' else None)),
                'age': to_num(diab_raw.get('age') or heart_raw.get('Age'), float),
                'hypertension': to_num(diab_raw.get('hypertension'), int),
                'heart_disease': to_num(diab_raw.get('heart_disease'), int),
                'smoking_history': diab_raw.get('smoking_history') or None,
                'bmi': to_num(diab_raw.get('bmi'), float),
                'HbA1c_level': to_num(diab_raw.get('HbA1c_level'), float),
                'blood_glucose_level': to_num(diab_raw.get('blood_glucose_level'), float),
            }])
            if diab_features:
                for c in diab_features:
                    if c not in diab_df.columns:
                        diab_df[c] = None
                diab_df = diab_df[diab_features]
            for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
                if col in diab_df.columns:
                    diab_df[col] = pd.to_numeric(diab_df[col], errors='coerce')

            # run predictions using the dedicated models
            try:
                cardio_prob = float(heart_model.predict_proba(heart_df)[:,1][0])
            except Exception:
                hprobs = heart_model.predict_proba(heart_df)
                cardio_prob = float(hprobs[0][:,1][0]) if isinstance(hprobs, list) else float(hprobs[:,1][0])

            try:
                dprobs = diab_model.predict_proba(diab_df)
                diabetes_prob = float(dprobs[:,1][0]) if not isinstance(dprobs, list) else float(dprobs[0][:,1][0])
            except Exception:
                dprobs = diab_model.predict_proba(diab_df)
                diabetes_prob = float(dprobs[0][:,1][0]) if isinstance(dprobs, list) else float(dprobs[:,1][0])

            result = {
                'cardio': {'prob': round(cardio_prob,4), 'risk': risk_band(cardio_prob)},
                'diabetes': {'prob': round(diabetes_prob,6), 'risk': risk_band(diabetes_prob)},
            }

            submitted = {'heart': heart_raw, 'diabetes': diab_raw}
        except Exception as e:
            print(f"Error: {e}")
            error = str(e)

    return render_template("combined_form.html", heart_fields=heart_fields, diabetes_fields=diabetes_fields, option_help=option_help, result=result, error=error, submitted=submitted)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for doctor diagnosis"""
    try:
        # Get form data
        heart_raw = {k: request.form.get(k, "").strip() for k in ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']}
        diab_raw = {k: request.form.get(k, "").strip() for k in ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']}

        def to_num(x, dtype=float):
            if x is None or x == "":
                return None
            try:
                return dtype(x)
            except Exception:
                return None

        # Build heart DataFrame
        heart_df = pd.DataFrame([{
            'Age': to_num(heart_raw.get('Age'), float),
            'Sex': heart_raw.get('Sex') or None,
            'ChestPainType': heart_raw.get('ChestPainType') or None,
            'RestingBP': to_num(heart_raw.get('RestingBP'), float),
            'Cholesterol': to_num(heart_raw.get('Cholesterol'), float),
            'FastingBS': to_num(heart_raw.get('FastingBS'), float),
            'RestingECG': heart_raw.get('RestingECG') or None,
            'MaxHR': to_num(heart_raw.get('MaxHR'), float),
            'ExerciseAngina': heart_raw.get('ExerciseAngina') or None,
            'Oldpeak': to_num(heart_raw.get('Oldpeak'), float),
            'ST_Slope': heart_raw.get('ST_Slope') or None,
        }])
        if heart_features:
            for c in heart_features:
                if c not in heart_df.columns:
                    heart_df[c] = None
            heart_df = heart_df[heart_features]
        for col in ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']:
            if col in heart_df.columns:
                heart_df[col] = pd.to_numeric(heart_df[col], errors='coerce')

        # Build diabetes DataFrame
        diab_df = pd.DataFrame([{
            'gender': (diab_raw.get('gender') or '').lower() or ('male' if heart_raw.get('Sex') == 'M' else ('female' if heart_raw.get('Sex') == 'F' else None)),
            'age': to_num(diab_raw.get('age') or heart_raw.get('Age'), float),
            'hypertension': to_num(diab_raw.get('hypertension'), int),
            'heart_disease': to_num(diab_raw.get('heart_disease'), int),
            'smoking_history': diab_raw.get('smoking_history') or None,
            'bmi': to_num(diab_raw.get('bmi'), float),
            'HbA1c_level': to_num(diab_raw.get('HbA1c_level'), float),
            'blood_glucose_level': to_num(diab_raw.get('blood_glucose_level'), float),
        }])
        if diab_features:
            for c in diab_features:
                if c not in diab_df.columns:
                    diab_df[c] = None
            diab_df = diab_df[diab_features]
        for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
            if col in diab_df.columns:
                diab_df[col] = pd.to_numeric(diab_df[col], errors='coerce')

        # Predictions
        try:
            cardio_prob = float(heart_model.predict_proba(heart_df)[:,1][0])
        except Exception:
            hprobs = heart_model.predict_proba(heart_df)
            cardio_prob = float(hprobs[0][:,1][0]) if isinstance(hprobs, list) else float(hprobs[:,1][0])

        try:
            dprobs = diab_model.predict_proba(diab_df)
            diabetes_prob = float(dprobs[:,1][0]) if not isinstance(dprobs, list) else float(dprobs[0][:,1][0])
        except Exception:
            dprobs = diab_model.predict_proba(diab_df)
            diabetes_prob = float(dprobs[0][:,1][0]) if isinstance(dprobs, list) else float(dprobs[:,1][0])

        return jsonify({
            'cardio': {'prob': round(cardio_prob,4), 'risk': risk_band(cardio_prob)},
            'diabetes': {'prob': round(diabetes_prob,6), 'risk': risk_band(diabetes_prob)},
        })
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)