const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const morgan = require('morgan');
const logger = require('./utils/logger');
const { errorHandler, notFoundHandler } = require('./middlewares/error.middleware');

// Import centralized routes
const routes = require('./routes');

const app = express();

// ==================== MIDDLEWARE CONFIGURATION ====================

// CORS
app.use(cors({
  origin: process.env.CORS_ORIGIN?.split(',') || '*',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

// Body parsers
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(bodyParser.json());

// Logging middleware
const morganFormat = ':method :url :status :response-time ms - :res[content-length]';
app.use(morgan(morganFormat, {
  skip: (req) => req.url === '/health', // Skip health check logs
  stream: {
    write: (message) => logger.info(message.trim()),
  },
}));

// Request logging (custom)
app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    const logData = {
      method: req.method,
      url: req.originalUrl,
      status: res.statusCode,
      duration: `${duration}ms`,
      ip: req.ip,
      userAgent: req.get('user-agent'),
    };
    
    if (res.statusCode >= 400) {
      logger.error('Request error:', logData);
    } else if (req.originalUrl !== '/health') {
      logger.info('Request:', logData);
    }
  });
  
  next();
});

// ==================== ROUTES ====================

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: '🏥 Clinic Management System API',
    version: '1.0.0',
    status: 'Running',
    timestamp: new Date().toISOString(),
    endpoints: {
      health: 'GET /health',
      api: 'GET /api',
      docs: 'GET /api-docs (coming soon)',
    },
    availableModules: [
      'auth',
      'users',
      'patients',
      'doctors',
      'appointments',
      'medical-records',
      'prescriptions',
      'medicines',
      'stocks',
      'timeslots',
      'schedules',
      'invoices',
      'lab-orders',
      'rooms',
      'suppliers',
      'notifications',
      'reports',
      'audit_logs',
    ],
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: process.env.NODE_ENV || 'development',
    database: 'connected', // You can add real DB health check here
    memory: {
      used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024) + ' MB',
      total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024) + ' MB',
    },
  });
});

// API info endpoint
app.get('/api', (req, res) => {
  res.json({
    message: 'Clinic Management API',
    version: '1.0.0',
    modules: {
      auth: '/api/auth',
      users: '/api/users',
      patients: '/api/patients',
      doctors: '/api/doctors',
      appointments: '/api/appointments',
      medicalRecords: '/api/medical-records',
      prescriptions: '/api/prescriptions',
      medicines: '/api/medicines',
      stocks: '/api/stocks',
      timeslots: '/api/timeslots',
      schedules: '/api/schedules',
      invoices: '/api/invoices',
      labOrders: '/api/lab-orders',
      rooms: '/api/rooms',
      suppliers: '/api/suppliers',
      notifications: '/api/notifications',
      reports: '/api/reports',
      auditLogs: '/api/audit_logs',
    },
  });
});

// Mount all API routes under /api prefix
app.use('/api', routes);

// ==================== ERROR HANDLING ====================

// 404 handler (must be before error handler)
app.use(notFoundHandler);

// Global error handler (must be last)
app.use(errorHandler);
module.exports = app;