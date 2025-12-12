-- Manual non-destructive migration: add missing columns used by application
-- Adds notified_at and notified_by to prescriptions, and prescription_id to invoices
ALTER TABLE public.prescriptions ADD COLUMN IF NOT EXISTS notified_at timestamptz;
ALTER TABLE public.prescriptions ADD COLUMN IF NOT EXISTS notified_by integer;
ALTER TABLE public.invoices ADD COLUMN IF NOT EXISTS prescription_id integer;

-- Note: Adding FK constraints is optional and may fail if existing values are invalid.
-- If you want to add a foreign key constraint and ensure integrity, run after verifying values:
-- ALTER TABLE public.invoices ADD CONSTRAINT invoices_prescription_id_fkey FOREIGN KEY (prescription_id) REFERENCES public.prescriptions(id) ON UPDATE NO ACTION;
