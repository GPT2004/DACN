const express = require('express');
const router = express.Router();
const prescriptionController = require('./prescriptions.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const {
  createPrescriptionSchema,
  updatePrescriptionSchema,
} = require('./prescriptions.validator');
const { ROLES } = require('../../config/constants');

// Get all prescriptions
router.get(
  '/',
  authenticate,
  prescriptionController.getPrescriptions
);

// (Get prescription by ID placed after more specific routes)

// Create prescription (Doctor only)
router.post(
  '/',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  validate(createPrescriptionSchema),
  prescriptionController.createPrescription
);

// Update prescription (Doctor only, DRAFT status)
router.put(
  '/:id',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  validate(updatePrescriptionSchema),
  prescriptionController.updatePrescription
);

// Approve prescription (Doctor only)
router.patch(
  '/:id/approve',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  prescriptionController.approvePrescription
);

// Notify reception (Doctor/Admin) - broadcast prescription ready for invoicing
router.post(
  '/:id/notify-reception',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  prescriptionController.notifyReception
);

// Get prescriptions waiting for invoicing (Receptionist)
router.get(
  '/for-invoicing',
  authenticate,
  authorize([ROLES.RECEPTIONIST, ROLES.ADMIN]),
  prescriptionController.getForInvoicing
);

// Dispense prescription (Pharmacist only)
const { dispensePrescriptionSchema } = require('./prescriptions.validator');
router.patch(
  '/:id/dispense',
  authenticate,
  authorize([ROLES.PHARMACIST, ROLES.ADMIN]),
  validate(dispensePrescriptionSchema),
  prescriptionController.dispensePrescription
);

// Download prescription PDF
router.get(
  '/:id/pdf',
  authenticate,
  prescriptionController.downloadPrescriptionPDF
);

// Get prescription by ID
router.get(
  '/:id',
  authenticate,
  prescriptionController.getPrescriptionById
);

// Delete prescription (Doctor/Admin, DRAFT only)
router.delete(
  '/:id',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  prescriptionController.deletePrescription
);

module.exports = router;
