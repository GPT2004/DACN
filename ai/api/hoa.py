# app_symptom_to_disease.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import os, sys, csv, joblib
from flask import Flask, request, render_template_string, jsonify

# ====== ĐƯỜNG DẪN ======
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "models"
UTILS_DIR  = ROOT / "scripts"
sys.path.insert(0, str(UTILS_DIR))

MODEL_PATH = MODELS_DIR / "symptom_xgb_model.pkl"
INFER_CFG  = MODELS_DIR / "inference_config.pkl"
DISEASE_MAP_CSV = ROOT / "data" / "diseases_to_department.csv"

# ====== UTIL CHUẨN HOÁ & RULES ======
from symptom_utils import vi_to_en_symptom_map, rule_override, get_common_diseases_vi

# ====== NẠP MODEL + CẤU HÌNH ======
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
infer_cfg = joblib.load(INFER_CFG) if INFER_CFG.exists() else {}
THRESHOLD = float(infer_cfg.get("THRESHOLD", 0.35))

CLASSES: List[str] = list(getattr(model, "classes_", []))

# ====== NẠP BẢN ĐỒ BỆNH (TUỲ CHỌN) ======
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

# ====== APP & TEMPLATE (tương tự nhưng hiển thị BỆNH) ======
app = Flask(__name__)

HTML = r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Triệu chứng → Gợi ý bệnh</title>
  <style>
    body{font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:900px;margin:32px auto;padding:0 16px;}
    h1{font-size:1.6rem;margin-bottom:8px} .muted{color:#666}
    form{margin:16px 0;padding:16px;border:1px solid #ddd;border-radius:12px}
    textarea{width:100%;min-height:90px;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:14px}
    button{padding:10px 16px;border:0;border-radius:10px;cursor:pointer;background:#111;color:#fff}
    .card{border:1px solid #eee;border-radius:12px;padding:14px;margin-top:16px}
    .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
    .chip{border:1px solid #ddd;border-radius:999px;padding:6px 10px;font-size:13px}
    table{border-collapse:collapse;width:100%;margin-top:8px}
    th,td{border-bottom:1px solid #eee;padding:8px;text-align:left;font-size:14px}
    .mono{font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace; font-size:13px}
  </style>
</head>
<body>
  <h1>Gợi ý bệnh từ triệu chứng</h1>
  <p class="muted">Nhập triệu chứng, cách nhau bởi dấu phẩy. Ví dụ: <span class="mono">sốt, ho khan, đau họng</span>.</p>

  <form method="post" action="/disease">
    <label for="sym">Triệu chứng</label>
    <textarea id="sym" name="symptoms" placeholder="ví dụ: sốt, ho, đau ngực, khó thở">{{ symptoms or "" }}</textarea>
    <div style="margin-top:8px">
      <button type="submit">Phân tích</button>
      <span class="muted">Ngưỡng: {{ threshold }}</span>
    </div>
  </form>

  {% if error %}
    <div class="card"><div style="color:#d33">{{ error }}</div></div>
  {% endif %}

  {% if result %}
    <div class="card">
      <h3>Gợi ý bệnh</h3>
      {% if result.diseases %}
        <ol>
        {% for d, s in result.diseases %}
          <li><strong>{{ d }}</strong> — {{ '{:.1f}%'.format(100*s) }}</li>
        {% endfor %}
        </ol>
      {% else %}
        <p>Không tìm thấy gợi ý.</p>
      {% endif %}

      <h4 style="margin-top:12px">Nguồn (khoa gợi ý)</h4>
      <ul>
        {% for dept, p in result.top_depts %}
          <li>{{ dept }} — {{ '{:.1f}%'.format(100*p) }}</li>
        {% endfor %}
      </ul>

      <details style="margin-top:10px">
        <summary class="muted">Chuỗi sau chuẩn hoá</summary>
        <div class="mono">{{ result.cleaned_text }}</div>
      </details>
    </div>
  {% endif %}
</body>
</html>
"""

# helper: clean and topk same as original
def _clean_input(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    mapped = vi_to_en_symptom_map(text)
    parts = [p.strip() for p in mapped.split(",") if p.strip()]
    return ", ".join(parts)

def _topk_probs(text: str, k: int = 5):
    if not getattr(model, "predict_proba", None):
        pred = model.predict([text])[0]
        return [(pred, 1.0)], 1.0, pred
    proba = model.predict_proba([text])[0]
    pairs = list(zip(CLASSES, proba))
    pairs.sort(key=lambda x: x[1], reverse=True)
    topk = pairs[:k]
    best_dept, best_prob = topk[0]
    return topk, best_prob, best_dept

def _aggregate_diseases_from_topk(topk, limit=12):
    # Build disease scores from department probabilities and disease ranks
    scores: Dict[str, float] = {}
    for dept, p in topk:
        diseases = get_common_diseases_vi(dept)
        for rank, d in enumerate(diseases[:15]):
            score = p * (1.0 / (rank + 1))
            scores[d] = scores.get(d, 0.0) + score
    # sort
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # normalize total to sum <=1 for display convenience
    total = sum(v for _, v in items) or 1.0
    normalized = [(name, val / total) for name, val in items[:limit]]
    return normalized

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML, symptoms='', result=None, error=None, threshold=THRESHOLD)

@app.route('/disease', methods=['GET','POST'])
def disease():
    if request.method == 'GET':
        return render_template_string(HTML, symptoms='', result=None, error=None, threshold=THRESHOLD)

    text = request.form.get('symptoms','')
    cleaned = _clean_input(text)
    if not cleaned:
        return render_template_string(HTML, symptoms=text, result=None, error='Vui lòng nhập ít nhất một triệu chứng.', threshold=THRESHOLD)

    rule_dept = rule_override(cleaned)
    if rule_dept:
        topk = [(rule_dept, 0.9)]
        diseases = _aggregate_diseases_from_topk(topk, limit=12)
        result = {'diseases': diseases, 'top_depts': [(to_vi := rule_dept, 0.9)], 'cleaned_text': cleaned}
        return render_template_string(HTML, symptoms=text, result=result, error=None, threshold=THRESHOLD)

    topk, maxp, best = _topk_probs(cleaned, k=6)
    diseases = _aggregate_diseases_from_topk(topk, limit=12)
    # convert department codes to Vietnamese names for display in source
    top_depts_vi = [( (lambda d: d)(d), p) for d, p in topk]
    result = {'diseases': diseases, 'top_depts': top_depts_vi, 'cleaned_text': cleaned}
    return render_template_string(HTML, symptoms=text, result=result, error=None, threshold=THRESHOLD)

@app.post('/api/disease')
def api_disease():
    data = request.get_json(silent=True) or {}
    text = data.get('symptoms','')
    cleaned = _clean_input(text)
    if not cleaned:
        return jsonify({'ok':False,'error':'missing symptoms'}), 400
    topk, maxp, best = _topk_probs(cleaned, k=6)
    diseases = _aggregate_diseases_from_topk(topk, limit=12)
    return jsonify({'ok':True,'cleaned':cleaned,'diseases': [{'d':d,'score':float(s)} for d,s in diseases],'topk': [{'dept':d,'prob':float(p)} for d,p in topk]})

if __name__ == '__main__':
    port = int(os.getenv('PORT','5010'))
    app.run(host='0.0.0.0', port=port, debug=True)
