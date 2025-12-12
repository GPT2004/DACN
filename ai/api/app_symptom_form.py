# app_symptom_form.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import os, sys, csv, joblib
from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS

# ====== ĐƯỜNG DẪN ======
ROOT = Path(__file__).resolve().parents[1]   # trỏ ra thư mục 'ai'
MODELS_DIR = ROOT / "data" / "models"
UTILS_DIR  = ROOT / "scripts"                # đảm bảo import đúng utils
sys.path.insert(0, str(UTILS_DIR))

MODEL_PATH = MODELS_DIR / "symptom_xgb_model.pkl"
INFER_CFG  = MODELS_DIR / "inference_config.pkl"
DISEASE_MAP_CSV = ROOT / "data" / "diseases_to_department.csv"   # (tuỳ chọn)

# ====== UTIL CHUẨN HOÁ & RULES ======
from symptom_utils import vi_to_en_symptom_map, rule_override, get_common_diseases_vi

# ====== NẠP MODEL + CẤU HÌNH ======
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)  # Pipeline(Tfidf -> LogisticRegression)
infer_cfg = joblib.load(INFER_CFG) if INFER_CFG.exists() else {}
THRESHOLD = float(infer_cfg.get("THRESHOLD", 0.20))      # hạ ngưỡng tự tin xuống 20% để hiện đa dạng
GENERAL_DEPT = str(infer_cfg.get("GENERAL_DEPT", "General Medicine"))

CLASSES: List[str] = list(getattr(model, "classes_", []))

# ====== ÁNH XẠ TÊN KHOA (EN → VI) ======
DEPT_VI_MAP: Dict[str, str] = {
    "General": "Nội tổng quát",
    "General Medicine": "Nội tổng quát",
    "Cardiology": "Tim mạch",
    "Gastroenterology": "Tiêu hóa",
    "Neurology": "Thần kinh",
    "ENT": "Tai Mũi Họng",
    "Dermatology": "Da liễu",
    "Orthopedics": "Chấn thương chỉnh hình",
    "Pulmonology": "Hô hấp",
    "Endocrinology": "Nội tiết",
    "Urology": "Tiết niệu",
    "Gynecology": "Phụ khoa",
    "Pediatrics": "Nhi khoa",
    "Ophthalmology": "Mắt",
    "Dentistry": "Răng hàm mặt",
    "Psychiatry": "Tâm thần",
}
def to_vi_dept(name: str) -> str:
    return DEPT_VI_MAP.get(str(name), str(name))

# ====== NẠP GỢI Ý BỆNH (TUỲ CHỌN) ======
DISEASE_BY_DEPT: Dict[str, List[str]] = {}
if DISEASE_MAP_CSV.exists():
    try:
        with open(DISEASE_MAP_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dpt = str(row.get("department", "")).strip()
                dis = str(row.get("disease", "")).strip()
                if dpt and dis:
                    DISEASE_BY_DEPT.setdefault(dpt, []).append(dis)
    except Exception:
        DISEASE_BY_DEPT = {}

# ====== APP ======
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:8080"]}})

INDEX_HTML = r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Triệu chứng → Gợi ý khoa khám</title>
  <style>
    body{font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:900px;margin:32px auto;padding:0 16px;}
    h1{font-size:1.6rem;margin-bottom:8px} .muted{color:#666}
    form{margin:16px 0;padding:16px;border:1px solid #ddd;border-radius:12px}
    textarea{width:100%;min-height:90px;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:14px}
    button{padding:10px 16px;border:0;border-radius:10px;cursor:pointer;background:#111;color:#fff}
    .row{display:flex; gap:16px; align-items:center; margin-top:8px}
    .card{border:1px solid #eee;border-radius:12px;padding:14px;margin-top:16px}
    .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
    .chip{border:1px solid #ddd;border-radius:999px;padding:6px 10px;font-size:13px}
    table{border-collapse:collapse;width:100%;margin-top:8px}
    th,td{border-bottom:1px solid #eee;padding:8px;text-align:left;font-size:14px}
    .ok{color:#0c7}
    .warn{color:#c70}
    .err{color:#d33}
    .mono{font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace; font-size:13px}
  </style>
</head>
<body>
  <h1>Gợi ý khoa khám từ triệu chứng</h1>
  <p class="muted">Nhập triệu chứng, cách nhau bởi dấu phẩy. Ví dụ: <span class="mono">sốt, ho khan, đau họng</span> hoặc
  <span class="mono">đau đầu, chóng mặt, buồn nôn</span>.</p>

  <form method="post" action="/symptom">
    <label for="sym">Triệu chứng</label>
    <textarea id="sym" name="symptoms" placeholder="ví dụ: sốt, ho, đau ngực, khó thở">{{ symptoms or "" }}</textarea>
    <div class="row">
      <button type="submit">Phân tích</button>
      <span class="muted">Ngưỡng tin cậy: {{ threshold }}</span>
    </div>
  </form>

  {% if error %}
    <div class="card"><div class="err">{{ error }}</div></div>
  {% endif %}

  {% if result %}
    <div class="card">
      <h3>Kết quả</h3>
      <p><b>Khoa gợi ý:</b> {{ result.suggested_dept }}
         {% if result.confidence is not none %}
         (<span class="{{ 'ok' if result.confidence >= threshold else 'warn' }}">
            độ tin cậy {{ '{:.1f}%'.format(100*result.confidence) }}
          </span>)
         {% endif %}
      </p>

      {% if result.topk %}
        <h4>Xác suất tất cả khoa</h4>
        <table>
          <thead><tr><th>Khoa</th><th>Xác suất</th><th>Bệnh thường gặp</th></tr></thead>
          <tbody>
          {% for item in result.topk %}
            <tr>
              <td>{{ item.dept }}</td>
              <td>{{ '{:.2f}%'.format(100*item.prob) }}</td>
              <td>
                {% if item.diseases %}
                  {% for d in item.diseases %}
                    <span class="chip">{{ d }}</span>
                  {% endfor %}
                {% else %}
                  Không có dữ liệu
                {% endif %}
              </td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      {% endif %}

      {% if result.alternatives and result.alternatives|length > 0 %}
        <h4>Lựa chọn khác (xác suất gần Top-1)</h4>
        <table>
          <thead><tr><th>Khoa</th><th>Xác suất</th></tr></thead>
          <tbody>
          {% for dept, prob in result.alternatives %}
            <tr><td>{{ dept }}</td><td>{{ '{:.2f}%'.format(100*prob) }}</td></tr>
          {% endfor %}
          </tbody>
        </table>
      {% endif %}

      {% if result.disease_suggestions %}
        <h4>Bệnh thường gặp (gợi ý)</h4>
        <div class="chips">
          {% for d in result.disease_suggestions %}
            <span class="chip">{{ d }}</span>
          {% endfor %}
        </div>
      {% endif %}

      <details style="margin-top:10px">
        <summary class="muted">Chuỗi sau chuẩn hoá</summary>
        <div class="mono">{{ result.cleaned_text }}</div>
      </details>
    </div>
  {% endif %}
</body>
</html>
"""

# các tham số gợi ý “lựa chọn khác”
NEAR_TIE_MARGIN = 0.12
ALT_MIN_PROB    = 0.20

def _clean_input(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    mapped = vi_to_en_symptom_map(text)      # VI → EN để khớp TF-IDF
    parts = [p.strip() for p in mapped.split(",") if p.strip()]
    return ", ".join(parts)

def _topk_probs(text: str, k: int = len(CLASSES)) -> Tuple[List[Dict], float, str]:
    if not getattr(model, "predict_proba", None):
        pred = model.predict([text])[0]
        dept_vi = to_vi_dept(pred)
        diseases = get_common_diseases_vi(pred)
        return [{"dept": dept_vi, "prob": 1.0, "diseases": diseases}], 1.0, pred
    proba = model.predict_proba([text])[0]
    pairs = list(zip(CLASSES, proba))
    pairs.sort(key=lambda x: x[1], reverse=True)
    best_dept, best_prob = pairs[0][0], pairs[0][1]
    topk_list = []
    for d, p in pairs[:k]:
        dept_vi = to_vi_dept(d)
        diseases = get_common_diseases_vi(d)
        topk_list.append({"dept": dept_vi, "prob": p, "diseases": diseases})
    return topk_list, best_prob, best_dept

def _suggest_diseases_for(dept: str, limit: int = 12) -> List[str]:
    return get_common_diseases_vi(dept)[:limit]

@app.route("/", methods=["GET"])
def home():
    return render_template_string(INDEX_HTML, symptoms="", result=None, error=None, threshold=THRESHOLD)

@app.route("/symptom", methods=["GET", "POST"])
def symptom():
    if request.method == "GET":
        return render_template_string(INDEX_HTML, symptoms="", result=None, error=None, threshold=THRESHOLD)

    text = request.form.get("symptoms", "")
    cleaned = _clean_input(text)
    if not cleaned:
        return render_template_string(INDEX_HTML, symptoms=text, result=None,
                                      error="Vui lòng nhập ít nhất một triệu chứng.", threshold=THRESHOLD)

    # Luôn chạy ML để có top-k đa dạng
    topk, maxp, best = _topk_probs(cleaned)

    # Nếu có rule_override, ưu tiên khoa từ rule
    rule_dept = rule_override(cleaned)
    show_alternatives = True  # Mặc định show topk
    
    if rule_dept:
      suggested_en = rule_dept
      confidence = 0.9
      show_alternatives = False  # Rule match = chắc chắn, chỉ show 1 khoa
    else:
      suggested_en = best if maxp >= THRESHOLD else GENERAL_DEPT
      confidence = maxp
      # Nếu model confidence >= 0.95, coi như chắc chắn, chỉ show 1 khoa
      if confidence >= 0.95:
        show_alternatives = False

    # Lựa chọn khác khi sát điểm
    alternatives_vi = []
    # bỏ alternatives vì hiển thị tất cả
    
    # Nếu không chắc chắn, lọc topk để show
    if not show_alternatives:
      topk = []  # Chỉ show khoa chính, không show xác suất các khoa khác

    diseases = get_common_diseases_vi(suggested_en)

    # map sang VI để hiển thị
    suggested_vi = to_vi_dept(suggested_en)
    # topk đã là list dict với dept vi

    result = {
      "suggested_dept": suggested_vi,
      "confidence": confidence,
      "topk": topk,
      "alternatives": alternatives_vi,
      "cleaned_text": cleaned,
      "disease_suggestions": diseases,
    }
    return render_template_string(INDEX_HTML, symptoms=text, result=result, error=None, threshold=THRESHOLD)

# ---- API JSON (tuỳ chọn) ----
@app.post("/api/symptom")
def api_symptom():
    data = request.get_json(silent=True) or {}
    text = data.get("symptoms", "")
    cleaned = _clean_input(text)
    if not cleaned:
        return jsonify({"ok": False, "error": "missing symptoms"}), 400

    # Luôn lấy topk từ ML
    topk, maxp, best = _topk_probs(cleaned)

    # Ưu tiên rule nếu có, ngược lại dùng ML
    rule_dept = rule_override(cleaned)
    show_alternatives = True  # Mặc định show topk
    
    if rule_dept:
      suggested_en = rule_dept
      confidence = 0.9
      show_alternatives = False  # Rule match = chắc chắn, chỉ show 1 khoa
    else:
      suggested_en = best if maxp >= THRESHOLD else GENERAL_DEPT
      confidence = maxp
      # Nếu model confidence >= 0.95, coi như chắc chắn, chỉ show 1 khoa
      if confidence >= 0.95:
        show_alternatives = False
    
    # Nếu không chắc chắn, lọc topk để show
    if not show_alternatives:
      topk = []  # Chỉ show khoa chính, không show xác suất các khoa khác

    diseases = get_common_diseases_vi(suggested_en)

    return jsonify({
        "ok": True,
        "threshold": THRESHOLD,
        "input": text,
        "cleaned": cleaned,
        "suggested_department_en": suggested_en,
        "suggested_department_vi": to_vi_dept(suggested_en),
        "confidence": float(confidence),
        "topk": topk,
        "diseases": diseases
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
