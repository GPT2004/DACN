-- Migration: add_patient_owner_and_fields
-- This migration file mirrors the SQL changes you applied manually.
-- It is safe to run even if columns/indexes already exist (uses IF NOT EXISTS checks).

BEGIN;

-- 1) Drop any UNIQUE index on patients.user_id (if exists)
DO $$
DECLARE
  idxname text;
BEGIN
  SELECT indexname INTO idxname
  FROM pg_indexes
  WHERE tablename = 'patients'
    AND indexdef ILIKE '%(user_id)%'
    AND indexdef ILIKE '%UNIQUE%';

  IF idxname IS NOT NULL THEN
    RAISE NOTICE 'Dropping unique index % on patients(user_id)', idxname;
    EXECUTE format('DROP INDEX IF EXISTS %I', idxname);
  ELSE
    RAISE NOTICE 'No explicit UNIQUE index on patients(user_id) found; skipping drop.';
  END IF;
END$$;

-- 2) Make user_id nullable
ALTER TABLE IF EXISTS patients
  ALTER COLUMN user_id DROP NOT NULL;

-- 3) Add owner_user_id column
ALTER TABLE IF EXISTS patients
  ADD COLUMN IF NOT EXISTS owner_user_id integer;

-- 4) Backfill owner_user_id with existing user_id where applicable
UPDATE patients
SET owner_user_id = user_id
WHERE owner_user_id IS NULL AND user_id IS NOT NULL;

-- 5) Add display fields if missing
ALTER TABLE IF EXISTS patients
  ADD COLUMN IF NOT EXISTS full_name varchar(255),
  ADD COLUMN IF NOT EXISTS phone varchar(50),
  ADD COLUMN IF NOT EXISTS email varchar(255),
  ADD COLUMN IF NOT EXISTS dob date;

-- 6) Add foreign key constraint for owner_user_id -> users(id) if not exists
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
    WHERE tc.table_name = 'patients'
      AND tc.constraint_type = 'FOREIGN KEY'
      AND kcu.column_name = 'owner_user_id'
  ) THEN
    ALTER TABLE patients
      ADD CONSTRAINT fk_patients_owner_user_id
        FOREIGN KEY (owner_user_id) REFERENCES users(id) ON UPDATE NO ACTION;
  ELSE
    RAISE NOTICE 'Foreign key on patients(owner_user_id) already exists; skipping.';
  END IF;
END$$;

-- 7) Create index on owner_user_id
CREATE INDEX IF NOT EXISTS idx_patients_owner_user_id ON patients(owner_user_id);

COMMIT;
