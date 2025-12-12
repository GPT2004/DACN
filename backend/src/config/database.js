/* eslint-disable no-console */
const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient({
  log: process.env.NODE_ENV === 'development' ? ['query', 'error', 'warn'] : ['error'],
});

// Test connection
prisma.$connect()
  .then(() => {
    try {
      const raw = process.env.DATABASE_URL || '';
      let host = 'unknown';
      let dbname = 'unknown';
      try {
        const m = raw.match(/@([^:\/]+)(?::(\d+))?\/?([^?]+)?/);
        if (m) {
          host = m[1];
          dbname = m[3] || 'unknown';
        }
      } catch (e) {
        // ignore
      }
      console.log(`✅ Database connected successfully (host=${host}, db=${dbname})`);
    } catch (err) {
      console.log('✅ Database connected successfully');
    }
  })
  .catch((err) => console.error('❌ Database connection failed:', err));

// Graceful shutdown
process.on('beforeExit', async () => {
  await prisma.$disconnect();
});

module.exports = prisma;