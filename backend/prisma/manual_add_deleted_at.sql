ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
ALTER TABLE IF EXISTS medicines ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
ALTER TABLE IF EXISTS stocks ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
