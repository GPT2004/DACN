-- Run this SQL directly against your Postgres database to add relations
-- between medical_records, prescriptions and lab_orders, and add exam_results
-- Note: this file intentionally uses direct SQL so it can be applied without prisma migrations.

BEGIN;

-- Add exam_results jsonb column to medical_records (store structured exam/examination results)
ALTER TABLE IF EXISTS medical_records
  ADD COLUMN IF NOT EXISTS exam_results jsonb;

-- Add medical_record_id to prescriptions
ALTER TABLE IF EXISTS prescriptions
  ADD COLUMN IF NOT EXISTS medical_record_id INT;
CREATE INDEX IF NOT EXISTS idx_prescriptions_medical_record ON prescriptions(medical_record_id);
-- Add foreign key constraint for prescriptions.medical_record_id if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'fk_prescriptions_medical_record'
  ) THEN
    ALTER TABLE prescriptions
      ADD CONSTRAINT fk_prescriptions_medical_record FOREIGN KEY (medical_record_id) REFERENCES medical_records(id) ON DELETE SET NULL ON UPDATE NO ACTION;
  END IF;
END$$;

-- Add medical_record_id to lab_orders
ALTER TABLE IF EXISTS lab_orders
  ADD COLUMN IF NOT EXISTS medical_record_id INT;
CREATE INDEX IF NOT EXISTS idx_laborders_medical_record ON lab_orders(medical_record_id);
-- Add foreign key constraint for lab_orders.medical_record_id if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'fk_laborders_medical_record'
  ) THEN
    ALTER TABLE lab_orders
      ADD CONSTRAINT fk_laborders_medical_record FOREIGN KEY (medical_record_id) REFERENCES medical_records(id) ON DELETE SET NULL ON UPDATE NO ACTION;
  END IF;
END$$;

COMMIT;

-- Usage notes:
-- 1) Apply this SQL to your Postgres DB (psql -f manual_add_medical_record_relations.sql)
-- 2) No Prisma migration is required; this script modifies DB directly.
