-- Migration: add_patient_profile_fields
-- Adds additional profile fields to `patients` to support expanded front-end form.

BEGIN;

ALTER TABLE IF EXISTS patients
  ADD COLUMN IF NOT EXISTS occupation varchar(255),
  ADD COLUMN IF NOT EXISTS id_type varchar(50),
  ADD COLUMN IF NOT EXISTS id_number varchar(100),
  ADD COLUMN IF NOT EXISTS nationality varchar(100),
  ADD COLUMN IF NOT EXISTS ethnicity varchar(100),
  ADD COLUMN IF NOT EXISTS old_province varchar(255),
  ADD COLUMN IF NOT EXISTS old_district varchar(255),
  ADD COLUMN IF NOT EXISTS old_ward varchar(255),
  ADD COLUMN IF NOT EXISTS old_street varchar(500),
  ADD COLUMN IF NOT EXISTS new_province varchar(255),
  ADD COLUMN IF NOT EXISTS new_district varchar(255),
  ADD COLUMN IF NOT EXISTS new_ward varchar(255),
  ADD COLUMN IF NOT EXISTS new_street varchar(500);

-- Optional indexes to speed queries by province/district
CREATE INDEX IF NOT EXISTS idx_patients_old_province ON patients(old_province);
CREATE INDEX IF NOT EXISTS idx_patients_new_province ON patients(new_province);

-- Additional identity and contact fields
ALTER TABLE IF EXISTS patients
  ADD COLUMN IF NOT EXISTS id_type varchar(50),
  ADD COLUMN IF NOT EXISTS id_number varchar(100),
  ADD COLUMN IF NOT EXISTS id_issue_date date,
  ADD COLUMN IF NOT EXISTS id_issue_place varchar(255),
  ADD COLUMN IF NOT EXISTS zalo boolean DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_patients_id_number ON patients(id_number);

COMMIT;
