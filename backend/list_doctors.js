const { PrismaClient } = require('@prisma/client');
const p = new PrismaClient();

(async () => {
  try {
    const docs = await p.doctors.findMany({ include: { user: true }, orderBy: { id: 'asc' } });
    console.log('Total doctors:', docs.length);
    docs.forEach(d => console.log(d.id, d.user.email, d.specialties));
  } catch (e) {
    console.error(e);
  } finally {
    await p.$disconnect();
  }
})();
