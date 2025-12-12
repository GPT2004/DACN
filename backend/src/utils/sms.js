const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const logger = require('./utils/logger');
const { errorHandler, notFoundHandler } = require('./middlewares/error.middleware');

// Import routes
const authRoutes = require('./modules/auth/auth.routes');
const usersRoutes = require('./modules/users/users.routes');
const patientsRoutes = require('./modules/patients/patients.routes');
const doctorsRoutes = require('./modules/doctors/doctors.routes');
const appointmentsRoutes = require('./modules/appointments/appointments.routes');
const medicalRecordsRoutes = require('./modules/medical-records/medical-records.routes');
const prescriptionsRoutes = require('./modules/prescriptions/prescriptions.routes');
const medicinesRoutes = require('./modules/medicines/medicines.routes');
const stocksRoutes = require('./modules/stocks/stocks.routes');
const timeslotsRoutes = require('./modules/timeslots/timeslots.routes');
const schedulesRoutes = require('./modules/schedules/schedules.routes');
const invoicesRoutes = require('./modules/invoices/invoices.routes');
const labOrdersRoutes = require('./modules/lab-orders/lab-orders.routes');
const roomsRoutes = require('./modules/rooms/rooms.routes');
const notificationsRoutes = require('./modules/notifications/notifications.routes');
const reportsRoutes = require('./modules/reports/reports.routes');
const aiRoutes = require('./modules/ai/ai.routes');

const app = express();

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN?.split(',') || '*',
  credentials: true,
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Logging
const morganFormat = ':method :url :status :response-time ms';
app.use(morgan(morganFormat, {
  stream: {
    write: (message) => logger.info(message.trim()),
  },
}));

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// API Routes
app.use('/api/auth', authRoutes);
app.use('/api/users', usersRoutes);
app.use('/api/patients', patientsRoutes);
app.use('/api/doctors', doctorsRoutes);
app.use('/api/appointments', appointmentsRoutes);
app.use('/api/medical-records', medicalRecordsRoutes);
app.use('/api/prescriptions', prescriptionsRoutes);
app.use('/api/medicines', medicinesRoutes);
app.use('/api/stocks', stocksRoutes);
app.use('/api/timeslots', timeslotsRoutes);
app.use('/api/schedules', schedulesRoutes);
app.use('/api/invoices', invoicesRoutes);
app.use('/api/lab-orders', labOrdersRoutes);
app.use('/api/rooms', roomsRoutes);
app.use('/api/notifications', notificationsRoutes);
app.use('/api/reports', reportsRoutes);
app.use('/api/ai', aiRoutes);

// 404 handler
app.use(notFoundHandler);

// Error handler (must be last)
app.use(errorHandler);

module.exports = app;