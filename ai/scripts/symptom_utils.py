import re

# ===== 1) Từ điển triệu chứng VI -> EN =====
VI2EN_SYMPTOMS = {
  # Tim mạch
  "đau ngực": "chest pain",
  "tức ngực": "chest tightness",
  "khó thở khi gắng sức": "shortness of breath on exertion",
  "khó thở": "shortness of breath",
  "đánh trống ngực": "palpitations",
  "hồi hộp": "palpitations",
  "choáng váng": "dizziness",
  "chóng mặt": "dizziness",
  "cao huyết áp": "hypertension",

  # Da liễu
  "ngứa": "itching",
  "ngứa da": "itching",
  "phát ban": "skin rash",
  "mề đay": "hives",
  "nổi mẩn": "rash",
  "mảng đỏ": "red patches",
  "khô da": "dry skin",
  "mụn": "acne",

  # Tai Mũi Họng
  "đau họng": "sore throat",
  "ngứa họng": "throat irritation",
  "viêm họng": "sore throat",
  "sổ mũi": "runny nose",
  "chảy mũi": "runny nose",
  "ngạt mũi": "stuffy nose",
  "nghẹt mũi": "stuffy nose",
  "viêm mũi": "nasal inflammation",
  "đau xoang": "sinus pressure",
  "viêm xoang": "sinus pressure",
  "đau tai": "ear pain",
  "ù tai": "tinnitus",
  "khàn tiếng": "hoarseness",

  # Hô hấp
  "ho khan": "dry cough",
  "ho có đờm": "productive cough",
  "ho": "cough",
  "khò khè": "wheezing",
  "đờm": "sputum",
  "thở khò khè": "wheezing",

  # Tiêu hoá
  "đau bụng": "abdominal pain",
  "đau dạ dày": "abdominal pain",
  "buồn nôn": "nausea",
  "nôn ói": "vomiting",
  "nôn": "vomiting",
  "tiêu chảy": "diarrhea",
  "ợ chua": "heartburn",
  "ợ nóng": "heartburn",
  "trào ngược": "acid reflux",
  "đầy hơi": "bloating",

  # Nội tiết
  "tiểu nhiều": "frequent urination",
  "khát nhiều": "excessive thirst",
  "sụt cân": "weight loss",
  "mệt mỏi": "fatigue",
  "run tay": "tremor"
}

# ===== 2) Ánh xạ khoa EN -> VI =====
DEPT_EN2VI = {
  "General Medicine": "Nội tổng quát",
  "Cardiology": "Tim mạch",
  "Gastroenterology": "Tiêu hóa",
  "Endocrinology": "Nội tiết",
  "Dermatology": "Da liễu",
  "ENT": "Tai Mũi Họng",
  "Pulmonology": "Hô hấp"
}

# ===== 2.1) Bệnh thường gặp cho mỗi khoa (VI) =====
DEPT_COMMON_DISEASES_VI = {
  "General Medicine": ["Sốt xuất huyết", "Thủy đậu", "Bệnh thương hàn", "Cảm lạnh thông thường", "Đau nửa đầu"],
  "Cardiology": ["Tăng huyết áp"],
  "Gastroenterology": ["Viêm gan D", "Viêm gan B", "Viêm gan E", "Viêm gan A", "Viêm gan do rượu"],
  "Endocrinology": ["Đái tháo đường", "Cường giáp", "Hạ đường huyết", "Suy giáp"],
  "Dermatology": ["Vẩy nến", "Mụn trứng cá"],
  "ENT": ["Nhồi máu cơ tim"],
  "Pulmonology": ["Hen phế quản", "Viêm phổi"]
}

# ===== 3) Rule-based: từ khoá đặc trưng cho từng khoa =====
SPECIALTY_KEYWORDS = {
    "Dermatology": [
        "itching","skin rash","rash","hives","red patches","eczema","lesion","blister","scaly","acne","dry skin"
    ],
    "Gastroenterology": [
        "abdominal pain","stomach pain","nausea","vomiting","diarrhea","heartburn","acid reflux","bloody stool","constipation","bloating"
    ],
    "Endocrinology": [
        "excessive thirst","frequent urination","polyuria","polydipsia","unexplained weight loss","weight loss","heat intolerance"
    ],
    "Cardiology": [
        "chest pain","chest tightness","tightness in chest","palpitations","hypertension","dizziness","shortness of breath on exertion"
    ],
    "ENT": [
        "sore throat","throat irritation","runny nose","sneezing","ear pain","tinnitus","nasal inflammation","nasal congestion","stuffy nose","sinus pressure","hoarseness"
    ],
    "Pulmonology": [
        "cough","dry cough","productive cough","wheezing","sputum","shortness of breath"
    ],
}

GENERAL_DEPT = "General Medicine"
THRESHOLD = 0.50

# ===== 4) Chuẩn hoá =====
def _basic_clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u00C0-\u1EF9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def vi_to_en_symptom_map(s: str) -> str:
    s = _basic_clean(s)
    for vi in sorted(VI2EN_SYMPTOMS.keys(), key=len, reverse=True):
        pattern = re.escape(vi)
        s = re.sub(rf"\b{pattern}\b", VI2EN_SYMPTOMS[vi], s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ===== 5) Rule override: tính điểm match =====
def rule_override(eng_text: str):
    t = eng_text.lower()
    best_dept, best_score = None, 0
    for dept, kws in SPECIALTY_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_dept, best_score = dept, score
    return best_dept if best_score > 0 else None

def translate_dept_to_vi(dept_en: str) -> str:
    return DEPT_EN2VI.get(dept_en, dept_en)

def get_common_diseases_vi(dept_en: str) -> list:
    return DEPT_COMMON_DISEASES_VI.get(dept_en, [])
