const express = require('express');
const router = express.Router();
const reportsController = require('./reports.controller');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');


router.use(authenticate);

// Dashboard statistics
router.get('/dashboard', 
  authorize(['Admin', 'Doctor']), 
  reportsController.getDashboardStats
);

// Revenue report
router.get('/revenue', 
  authorize(['Admin']), 
  reportsController.getRevenueReport
);

// Appointment report
router.get('/appointments', 
  authorize(['Admin', 'Receptionist']), 
  reportsController.getAppointmentReport
);

// Doctor performance
router.get('/doctors/performance', 
  authorize(['Admin']), 
  reportsController.getDoctorPerformance
);

// Common diseases
router.get('/diseases/common', 
  authorize(['Admin', 'Doctor']), 
  reportsController.getCommonDiseases
);

// Stock report
router.get('/stock', 
  authorize(['Admin', 'Pharmacist']), 
  reportsController.getStockReport
);

// Patient report
router.get('/patients', 
  authorize(['Admin']), 
  reportsController.getPatientReport
);

// Export report
router.get('/export', 
  authorize(['Admin']), 
  reportsController.exportReport
);

module.exports = router;
