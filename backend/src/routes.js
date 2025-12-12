const express = require('express');
const router = express.Router();

// Import all module routes
const authRoutes = require('./modules/auth/auth.routes');
const usersRoutes = require('./modules/users/users.routes');
const patientsRoutes = require('./modules/patients/patients.routes');
const doctorsRoutes = require('./modules/doctors/doctors.routes');
const appointmentsRoutes = require('./modules/appointments/appointments.routes');
const medicalRecordsRoutes = require('./modules/medical-records/medical_records.routes');
const prescriptionsRoutes = require('./modules/prescriptions/prescriptions.routes');
const medicinesRoutes = require('./modules/medicines/medicines.routes');
const stocksRoutes = require('./modules/stocks/stocks.routes');
const timeslotsRoutes = require('./modules/timeslots/timeslots.routes');
const schedulesRoutes = require('./modules/schedules/schedules.routes');
const invoicesRoutes = require('./modules/invoices/invoices.routes');
const labOrdersRoutes = require('./modules/lab-orders/lab_orders.routes');
const roomsRoutes = require('./modules/rooms/rooms.routes');
const suppliersRoutes = require('./modules/suppliers/suppliers.routes');
const notificationsRoutes = require('./modules/notifications/notifications.routes');
const reportsRoutes = require('./modules/reports/reports.routes');
const auditLogsRoutes = require('./modules/audit-logs/audit-logs.routes');
const reviewsRoutes = require('./modules/reviews/reviews.routes');
const specialtiesRoutes = require('./modules/specialties/specialties.routes');
const aiRoutes = require('./modules/ai/ai.routes');

router.get('/health', (req, res) => {
  res.json({ 
    success: true,
    status: 'OK', 
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Mount all routes
router.use('/auth', authRoutes);
router.use('/users', usersRoutes);
router.use('/patients', patientsRoutes);
router.use('/doctors', doctorsRoutes);
router.use('/appointments', appointmentsRoutes);
router.use('/medical-records', medicalRecordsRoutes);
router.use('/prescriptions', prescriptionsRoutes);
router.use('/medicines', medicinesRoutes);
router.use('/stocks', stocksRoutes);
router.use('/timeslots', timeslotsRoutes);
router.use('/schedules', schedulesRoutes);
router.use('/invoices', invoicesRoutes);
router.use('/lab-orders', labOrdersRoutes);
router.use('/rooms', roomsRoutes);
router.use('/suppliers', suppliersRoutes);
router.use('/notifications', notificationsRoutes);
router.use('/reports', reportsRoutes);
router.use('/audit-logs', auditLogsRoutes);
router.use('/reviews', reviewsRoutes);
router.use('/specialties', specialtiesRoutes);
router.use('/ai', aiRoutes);

module.exports = router;
