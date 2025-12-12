-- Non-destructive migration: add prescription_id to invoices and FK if missing
BEGIN;

-- Add column if not exists
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'invoices' AND column_name = 'prescription_id'
  ) THEN
    ALTER TABLE public.invoices ADD COLUMN prescription_id integer;
  END IF;
END;
$$;

-- Add foreign key constraint safely
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'invoices_prescription_id_fkey'
  ) THEN
    ALTER TABLE public.invoices
      ADD CONSTRAINT invoices_prescription_id_fkey
      FOREIGN KEY (prescription_id)
      REFERENCES public.prescriptions (id)
      ON UPDATE NO ACTION
      ON DELETE SET NULL;
  END IF;
END;
$$;

COMMIT;