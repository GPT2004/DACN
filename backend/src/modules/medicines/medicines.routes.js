const express = require('express');
const router = express.Router();
const medicineController = require('./medicines.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const {
  createMedicineSchema,
  updateMedicineSchema,
  createStockSchema,
  updateStockSchema,
  importMedicinesSchema,
} = require('./medicines.validator');
const { ROLES } = require('../../config/constants');

// Medicine routes
router.get(
  '/',
  authenticate,
  medicineController.getMedicines
);

router.get(
  '/:id',
  authenticate,
  medicineController.getMedicineById
);

router.post(
  '/',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  validate(createMedicineSchema),
  medicineController.createMedicine
);

router.put(
  '/:id',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  validate(updateMedicineSchema),
  medicineController.updateMedicine
);

router.delete(
  '/:id',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  medicineController.deleteMedicine
);

router.post(
  '/:id/restore',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  medicineController.restoreMedicine
);

router.post(
  '/import',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  validate(importMedicinesSchema),
  medicineController.importMedicines
);

// Get total stock for a medicine (must be before /stocks/ routes)
router.get(
  '/:id/stock',
  authenticate,
  medicineController.getStockByMedicineId
);

// Stock routes
router.get(
  '/stocks/all',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST, ROLES.DOCTOR]),
  medicineController.getStocks
);

router.get(
  '/stocks/summary',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  medicineController.getStockSummary
);

router.get(
  '/stocks/low-stock',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  medicineController.getLowStockAlerts
);

router.get(
  '/stocks/expiring',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  medicineController.getExpiringMedicines
);

router.get(
  '/stocks/:id',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  medicineController.getStockById
);

router.post(
  '/stocks',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  validate(createStockSchema),
  medicineController.createStock
);

router.put(
  '/stocks/:id',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  validate(updateStockSchema),
  medicineController.updateStock
);

router.delete(
  '/stocks/:id',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.PHARMACIST]),
  medicineController.deleteStock
);

module.exports = router;
