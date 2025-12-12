-- Create index for better query performance (constraint already exists in DB)
CREATE INDEX IF NOT EXISTS idx_timeslots_doctor_date ON public.timeslots(doctor_id, date);
CREATE INDEX IF NOT EXISTS idx_timeslots_active ON public.timeslots(is_active);
