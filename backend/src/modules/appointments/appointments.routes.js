const express = require('express');
const router = express.Router();
const appointmentController = require('./appointments.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const { 
  createAppointmentSchema, 
  updateAppointmentSchema,
  cancelAppointmentSchema,
  rescheduleAppointmentSchema
} = require('./appointments.validator');
const { ROLES } = require('../../config/constants');

// Get my appointments (current logged-in patient)
router.get(
  '/me',
  authenticate,
  appointmentController.getMyAppointments
);

// Get all appointments for patient (no pagination)
router.get(
  '/all/all-appointments',
  authenticate,
  appointmentController.getAllAppointmentsByPatient
);

// Get appointment history (past appointments for current patient)
router.get(
  '/history/list',
  authenticate,
  appointmentController.getAppointmentHistory
);

// Get all appointments (with filters)
router.get(
  '/',
  authenticate,
  appointmentController.getAppointments
);

// Get appointment by ID
router.get(
  '/:id',
  authenticate,
  appointmentController.getAppointmentById
);

// Create new appointment
router.post(
  '/',
  authenticate,
  authorize([ROLES.PATIENT, ROLES.RECEPTIONIST, ROLES.ADMIN]),
  validate(createAppointmentSchema),
  appointmentController.createAppointment
);

// Update appointment
router.put(
  '/:id',
  authenticate,
  authorize([ROLES.PATIENT, ROLES.RECEPTIONIST, ROLES.ADMIN]),
  validate(updateAppointmentSchema),
  appointmentController.updateAppointment
);

// Confirm appointment (Receptionist/Admin)
router.patch(
  '/:id/confirm',
  authenticate,
  authorize([ROLES.RECEPTIONIST, ROLES.ADMIN]),
  appointmentController.confirmAppointment
);

// Check-in appointment (Receptionist)
router.patch(
  '/:id/check-in',
  authenticate,
  authorize([ROLES.RECEPTIONIST, ROLES.ADMIN]),
  appointmentController.checkInAppointment
);

// Start appointment (Doctor)
router.patch(
  '/:id/start',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  appointmentController.startAppointment
);

// Complete appointment (Doctor)
router.patch(
  '/:id/complete',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  appointmentController.completeAppointment
);

// Mark as no-show (Receptionist/Admin)
router.patch(
  '/:id/no-show',
  authenticate,
  authorize([ROLES.RECEPTIONIST, ROLES.ADMIN]),
  appointmentController.markNoShow
);

// Cancel appointment
router.patch(
  '/:id/cancel',
  authenticate,
  validate(cancelAppointmentSchema),
  appointmentController.cancelAppointment
);

// Patient confirms they will attend (notify reception)
router.patch(
  '/:id/patient-confirm',
  authenticate,
  authorize([ROLES.PATIENT]),
  appointmentController.patientConfirmAppointment
);

// Public confirm via token (from email link)
router.get(
  '/:id/confirm',
  appointmentController.confirmByToken
);

// Reschedule appointment
router.patch(
  '/:id/reschedule',
  authenticate,
  validate(rescheduleAppointmentSchema),
  appointmentController.rescheduleAppointment
);

// Delete appointment (permanent)
router.delete(
  '/:id',
  authenticate,
  authorize([ROLES.RECEPTIONIST, ROLES.ADMIN]),
  appointmentController.deleteAppointment
);

module.exports = router;