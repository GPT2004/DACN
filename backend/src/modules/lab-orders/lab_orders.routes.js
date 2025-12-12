const express = require('express');
const router = express.Router();
const labOrderController = require('./lab_orders.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const {
  createLabOrderSchema,
  updateLabOrderSchema,
  updateLabResultsSchema,
} = require('./lab_orders.validator');
const { ROLES } = require('../../config/constants');

// Get all lab orders
router.get(
  '/',
  authenticate,
  labOrderController.getLabOrders
);

// Get lab order by ID
router.get(
  '/:id',
  authenticate,
  labOrderController.getLabOrderById
);

// Create lab order (Doctor only)
router.post(
  '/',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  validate(createLabOrderSchema),
  labOrderController.createLabOrder
);

// Update lab order (Doctor only, PENDING status)
router.put(
  '/:id',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  validate(updateLabOrderSchema),
  labOrderController.updateLabOrder
);

// Update lab results (Lab Tech/Admin)
router.patch(
  '/:id/results',
  authenticate,
  authorize([ROLES.LAB_TECH, ROLES.ADMIN]),
  validate(updateLabResultsSchema),
  labOrderController.updateLabResults
);

// Complete lab order (Lab Tech/Admin)
router.patch(
  '/:id/complete',
  authenticate,
  authorize([ROLES.LAB_TECH, ROLES.ADMIN]),
  labOrderController.completeLabOrder
);

// Delete lab order (Doctor/Admin, PENDING only)
router.delete(
  '/:id',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  labOrderController.deleteLabOrder
);

module.exports = router;
