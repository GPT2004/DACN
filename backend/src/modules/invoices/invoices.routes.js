const express = require('express');
const router = express.Router();
const invoiceController = require('./invoices.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const {
  createInvoiceSchema,
  updateInvoiceSchema,
  payInvoiceSchema,
  refundInvoiceSchema,
} = require('./invoices.validator');
const { ROLES } = require('../../config/constants');

// Get all invoices
router.get(
  '/',
  authenticate,
  invoiceController.getInvoices
);

// Get invoices by current patient
router.get(
  '/by-patient',
  authenticate,
  invoiceController.getInvoicesByPatient
);

// Get revenue summary
router.get(
  '/reports/revenue',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.RECEPTIONIST]),
  invoiceController.getRevenueSummary
);

// (Get invoice by ID route placed after more specific routes)

// Create invoice (Receptionist/Admin)
router.post(
  '/',
  authenticate,
  authorize([ROLES.RECEPTIONIST, ROLES.ADMIN]),
  validate(createInvoiceSchema),
  invoiceController.createInvoice
);

// Create invoice from prescription (Receptionist/Admin)
router.post(
  '/from-prescription',
  authenticate,
  authorize([ROLES.RECEPTIONIST, ROLES.ADMIN]),
  invoiceController.createFromPrescription
);

// Download invoice PDF
router.get(
  '/:id/pdf',
  authenticate,
  invoiceController.downloadInvoicePDF
);

// Get invoice by ID
router.get(
  '/:id',
  authenticate,
  invoiceController.getInvoiceById
);

// Update invoice
router.put(
  '/:id',
  authenticate,
  authorize([ROLES.RECEPTIONIST, ROLES.ADMIN]),
  validate(updateInvoiceSchema),
  invoiceController.updateInvoice
);

// Pay invoice
router.patch(
  '/:id/pay',
  authenticate,
  authorize([ROLES.RECEPTIONIST, ROLES.ADMIN]),
  validate(payInvoiceSchema),
  invoiceController.payInvoice
);

// Refund invoice
router.patch(
  '/:id/refund',
  authenticate,
  authorize([ROLES.ADMIN]),
  validate(refundInvoiceSchema),
  invoiceController.refundInvoice
);

// Delete invoice (Admin only)
router.delete(
  '/:id',
  authenticate,
  authorize([ROLES.ADMIN]),
  invoiceController.deleteInvoice
);

module.exports = router;
