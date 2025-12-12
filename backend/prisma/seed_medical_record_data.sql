-- Seed a medical_record and linked prescription + lab_order
BEGIN;

WITH new_rec AS (
  INSERT INTO medical_records (appointment_id, patient_id, doctor_id, diagnosis, notes, exam_results, attachments, created_at, updated_at)
  VALUES (
    1,
    1,
    1,
    'Hypertension',
    'Patient reports headaches. BP elevated.',
    $$
    {"blood_pressure": "150/95", "heart_rate": 88}
    $$::jsonb,
    $$[]$$::jsonb,
    now(), now()
  )
  RETURNING id
)
INSERT INTO prescriptions (appointment_id, medical_record_id, doctor_id, patient_id, items, total_amount, status, created_at, updated_at)
SELECT
  1,
  id,
  1,
  1,
  $$
  [{"medicine_id": 1, "medicine_name": "Paracetamol", "quantity": 10, "unit_price": 10000, "instructions": "Take after meals", "dosage": "500mg"}]
  $$::jsonb,
  100000,
  'DRAFT',
  now(), now()
FROM new_rec;

-- Insert a lab order linked to the same medical record
WITH rec AS (
  SELECT id FROM medical_records ORDER BY id DESC LIMIT 1
)
INSERT INTO lab_orders (appointment_id, medical_record_id, patient_id, doctor_id, tests, status, created_at, updated_at)
SELECT
  1,
  id,
  1,
  1,
  $$
  [{"test_code": "CBC", "name": "Complete Blood Count"}, {"test_code": "GLU", "name": "Glucose"}]
  $$::jsonb,
  'PENDING',
  now(), now()
FROM rec;

COMMIT;
