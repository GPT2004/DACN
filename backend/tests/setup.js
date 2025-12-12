/* eslint-disable no-console */
const { PrismaClient } = require('@prisma/client');
const bcrypt = require('bcryptjs');

const prisma = new PrismaClient();

// Setup before all tests
beforeAll(async () => {
  console.log('Setting up test database...');
  
  // Clean database
  await cleanDatabase();
  
  // Seed test data
  await seedTestData();
  
  console.log('Test database ready');
});

// Cleanup after all tests
afterAll(async () => {
  console.log('Cleaning up test database...');
  await cleanDatabase();
  await prisma.$disconnect();
  console.log('Test cleanup completed');
});

// Clean database
async function cleanDatabase() {
  const tables = [
    'audit_logs',
    'notifications',
    'lab_orders',
    'invoices',
    'prescriptions',
    'medical_records',
    'appointments',
    'timeslots',
    'schedules',
    'stocks',
    'medicines',
    'rooms',
    'suppliers',
    'patients',
    'doctors',
    'users',
    'roles',
  ];

  for (const table of tables) {
    await prisma.$executeRawUnsafe(`TRUNCATE TABLE ${table} CASCADE`);
  }
}

// Seed test data
async function seedTestData() {
  const hashedPassword = await bcrypt.hash('test123', 10);

  // Create roles
  const adminRole = await prisma.roles.create({
    data: { name: 'Admin', description: 'Administrator' },
  });

  const doctorRole = await prisma.roles.create({
    data: { name: 'Doctor', description: 'Doctor' },
  });

  const patientRole = await prisma.roles.create({
    data: { name: 'Patient', description: 'Patient' },
  });

  await prisma.roles.create({
    data: { name: 'Receptionist', description: 'Receptionist' },
  }); 

  // Create test users
  const adminUser = await prisma.users.create({
    data: {
      email: 'admin.test@clinic.com',
      password: hashedPassword,
      full_name: 'Test Admin',
      phone: '0900000001',
      role_id: adminRole.id,
    },
  });

  const doctorUser = await prisma.users.create({
    data: {
      email: 'doctor.test@clinic.com',
      password: hashedPassword,
      full_name: 'Test Doctor',
      phone: '0900000002',
      role_id: doctorRole.id,
    },
  });

  const patientUser = await prisma.users.create({
    data: {
      email: 'patient.test@clinic.com',
      password: hashedPassword,
      full_name: 'Test Patient',
      phone: '0900000003',
      dob: new Date('1990-01-01'),
      role_id: patientRole.id,
    },
  });

  // Create doctor profile
  global.testDoctor = await prisma.doctors.create({
    data: {
      user_id: doctorUser.id,
      license_number: 'TEST-LIC-001',
      specialties: 'Nội khoa',
      consultation_fee: 150000,
    },
  });

  // Create patient profile
  global.testPatient = await prisma.patients.create({
    data: {
      user_id: patientUser.id,
      gender: 'Male',
      blood_type: 'A+',
    },
  });

  // Create room
  global.testRoom = await prisma.rooms.create({
    data: {
      name: 'Test Room 1',
      type: 'Examination',
      capacity: 1,
    },
  });

  // Create medicine
  global.testMedicine = await prisma.medicines.create({
    data: {
      name: 'Test Medicine',
      code: 'TEST-MED-001',
      unit: 'Viên',
      price: 5000,
    },
  });

  // Store test credentials
  global.testCredentials = {
    admin: { email: 'admin.test@clinic.com', password: 'test123' },
    doctor: { email: 'doctor.test@clinic.com', password: 'test123' },
    patient: { email: 'patient.test@clinic.com', password: 'test123' },
  };

  global.testUsers = {
    admin: adminUser,
    doctor: doctorUser,
    patient: patientUser,
  };
}

module.exports = { prisma };