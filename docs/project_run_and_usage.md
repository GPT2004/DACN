# Project Run and Usage Guide

## 1) Prerequisites
- Node.js >= 18, npm
- Python 3.9+ with pip
- PostgreSQL running (configure .env for backend)
- Ports: frontend 3000, backend 8080, AI services 5000/5001

## 2) Setup
### Backend
1. `cd backend`
2. Copy `.env.example` to `.env`, set DB URL.
3. `npm install`
4. Khởi tạo DB (tùy chọn) từ file SQL: `psql -f clinic.sql` hoặc import qua client GUI.
5. `npx prisma migrate deploy`
6. `npm run seed` (optional sample data)
7. `npm run dev` (or `npm start`)

### Frontend
1. `cd frontend`
2. `npm install`
3. Create `.env` with `VITE_API_URL=http://localhost:8080/api`
4. `npm run dev`

### AI Services
1. `cd ai`
2. (Optional) create venv: `python -m venv .venv && .venv\Scripts\activate`
3. `pip install -r requirements.txt` (file nằm tại `ai/requirements.txt`)
   - Thư viện chính: `flask`, `flask-cors`, `scikit-learn`, `xgboost`, `pandas`, `joblib`, `numpy`.
4. Run services:
   - `python api/app_symptom_form.py` (port 5001)
   - `python api/app_combined_form.py` (port 5000) (or cardio/diabetes variants)

## 3) Usage
- Login with appropriate role (Doctor, Receptionist, Pharmacist, Patient).
- Doctor: create medical records, prescribe via modal, send to patient; AI suggestions available on forms.
- Receptionist: prescriptions inbox, invoices, check-in, schedule.
- Pharmacist: manage medicines (import Excel/CSV), edit, add stock, see expiring/low stock.
- Patient: view records, prescriptions.

## 4) Import Medicines
- Sample files: `frontend/public/sample_medicine_import.csv`, `sample_medicine_import_extra.csv`, `sample_medicine_import_extra2.csv`.
- Required columns: name, code, description, unit, price, batch_number, expiry_date (YYYY-MM-DD), quantity.
- Import: Pharmacist UI -> “Import Excel” -> select CSV/XLSX.
- Behavior: match by code, else name (case-insensitive). If exists: add/update stock; if new: create medicine + stock. Duplicate batch merges quantity.

## 5) Environment Variables (backend)
- `DATABASE_URL`: PostgreSQL connection.
- `PORT`: default 8080.
- Mail/SMS keys if notifications used.

## 6) Testing
- Backend: `npm test` (if configured), `npm run lint`.
- Frontend: `npm run test`, `npm run lint`.
- AI: run scripts and hit endpoints with curl/Postman to verify JSON responses.

## 7) Build/Deploy
- Frontend build: `npm run build` -> `dist/`.
- Backend: run with process manager (PM2) or container; ensure prisma migrations applied.
- Configure CORS to allow frontend origin.

## 8) Troubleshooting
- Import shows success but no new rows: check name/code duplicates; list may merge into existing items; refresh page.
- Prescription not in reception inbox: ensure doctor calls notifyReception (handled after create). Refresh `/receptionist/prescriptions-inbox`.
- AI errors: ensure ports 5000/5001 reachable; CORS enabled.
- DB errors: re-check `DATABASE_URL`, run migrations.

## 9) Key Paths
- Backend API: `backend/src/modules` (prescriptions, medical_records, medicines...).
- Frontend pages: `frontend/src/pages` (doctor, receptionist, pharmacist, patient).
- AI services: `ai/api/*.py`, models in `ai/data/models/`.
- Samples: `frontend/public/sample_medicine_import*.csv`.
