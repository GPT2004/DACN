-- Migration: create_pending_registrations
-- Stores pending registrations until email is confirmed

CREATE TABLE IF NOT EXISTS "pending_registrations" (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  full_name VARCHAR(255),
  password_hash TEXT NOT NULL,
  data JSONB DEFAULT '{}',
  token VARCHAR(255) NOT NULL UNIQUE,
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
