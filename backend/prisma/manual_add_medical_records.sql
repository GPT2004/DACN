-- prisma/manual_add_medical_records.sql
-- Run this directly against your Postgres database (psql) to add a medical_records table

CREATE TABLE IF NOT EXISTS medical_records (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER NOT NULL,
  doctor_id INTEGER NULL,
  title VARCHAR(255) NULL,
  diagnosis TEXT NULL,
  notes TEXT NULL,
  attachments JSONB NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Optional: foreign key constraints if users table is named 'users' and doctors are users
-- ALTER TABLE medical_records ADD CONSTRAINT fk_patient_user FOREIGN KEY (patient_id) REFERENCES users(id);
-- ALTER TABLE medical_records ADD CONSTRAINT fk_doctor_user FOREIGN KEY (doctor_id) REFERENCES users(id);

-- index for faster lookup by patient
CREATE INDEX IF NOT EXISTS idx_medical_records_patient ON medical_records(patient_id);
CREATE INDEX IF NOT EXISTS idx_medical_records_doctor ON medical_records(doctor_id);
