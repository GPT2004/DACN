-- Migration: create_shared_medical_records
-- Creates a lightweight table recording which medical records were shared to which patient rows
-- Run this SQL in your database (psql) or integrate into your migration workflow

CREATE TABLE IF NOT EXISTS "shared_medical_records" (
  id SERIAL PRIMARY KEY,
  medical_record_id INTEGER NOT NULL REFERENCES medical_records(id) ON DELETE CASCADE,
  recipient_patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  shared_by_user_id INTEGER NULL REFERENCES users(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Prevent duplicate shares for same recipient
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'shared_medical_records_unique_idx'
  ) THEN
    CREATE UNIQUE INDEX shared_medical_records_unique_idx ON public.shared_medical_records (medical_record_id, recipient_patient_id);
  END IF;
END$$;
