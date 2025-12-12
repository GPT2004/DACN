const express = require('express');
const router = express.Router();
const auditLogsController = require('./audit_logs.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const { createAuditLogSchema } = require('./audit_logs.validator');

router.use(authenticate);

// Get all audit logs (Admin only)
router.get('/', 
  authorize(['Admin']), 
  auditLogsController.getAllAuditLogs
);

// Get audit log by ID (Admin only)
router.get('/:id', 
  authorize(['Admin']), 
  auditLogsController.getAuditLogById
);

// Get action types (Admin only)
router.get('/actions/types', 
  authorize(['Admin']), 
  auditLogsController.getActionTypes
);

// Get user's audit logs (Admin only)
router.get('/user/:userId', 
  authorize(['Admin']), 
  auditLogsController.getUserAuditLogs
);

// Create audit log (Admin only - for manual logging)
router.post('/', 
  authorize(['Admin']), 
  validate(createAuditLogSchema), 
  auditLogsController.createAuditLog
);

// Delete old logs (Admin only)
router.delete('/cleanup', 
  authorize(['Admin']), 
  auditLogsController.deleteOldLogs
);

module.exports = router;