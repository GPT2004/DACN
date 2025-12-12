const express = require('express');
const router = express.Router();
const patientsController = require('./patients.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');
const { uploadSingle, handleUploadError } = require('../../middlewares/upload.middleware');

const { createPatientSchema, updatePatientSchema, updateProfileSchema, createDependentPatientSchema, createDependentPatientExtendedSchema, updateProfileExtendedSchema } = require('./patients.validator');
const { createPatientQuickSchema } = require('./patients.validator');

router.use(authenticate);

// Get all patients
router.get('/', authorize(['Admin', 'Receptionist', 'Doctor']), patientsController.getAllPatients);

// Get patient by ID
router.get('/:id', authorize(['Admin', 'Receptionist', 'Doctor', 'Patient']), patientsController.getPatientById);

// Get patient medical records
router.get('/:id/medical-records', authorize(['Admin', 'Doctor', 'Patient']), patientsController.getPatientMedicalRecords);

// Get patient appointments
router.get('/:id/appointments', authorize(['Admin', 'Receptionist', 'Doctor', 'Patient']), patientsController.getPatientAppointments);

// Get patient prescriptions
router.get('/:id/prescriptions', authorize(['Admin', 'Doctor', 'Patient']), patientsController.getPatientPrescriptions);

// Get patient invoices
router.get('/:id/invoices', authorize(['Admin', 'Receptionist', 'Patient']), patientsController.getPatientInvoices);

// Get my profile (Patient)
router.get('/me/profile', authorize(['Patient']), patientsController.getMyProfile);

// Debug: get shared medical records for current patient user
router.get('/me/shared-medical-records', authorize(['Patient']), patientsController.getMySharedMedicalRecords);

// Update my profile (Patient) - allow extended profile fields (occupation, id_type, etc.)
router.put('/me/profile', authorize(['Patient']), uploadSingle('avatar_url'), validate(updateProfileExtendedSchema), patientsController.updateMyProfile);

// Get my patients (own profiles and dependents)
router.get('/me/list', authorize(['Patient']), patientsController.getMyPatients);

// Delete my dependent profile (owner)
router.delete('/me/:id', authorize(['Patient']), patientsController.deleteMyDependent);

// Delete my own profile (Patient) - deletes user and patient record
router.delete('/me/profile', authorize(['Patient']), patientsController.deleteMyProfile);

// Create dependent patient profile for current patient user
router.post('/me', authorize(['Patient']), validate(createDependentPatientExtendedSchema), patientsController.createDependentPatient);

// Update a dependent or owned patient profile (Patient owner)
router.put('/me/:id', authorize(['Patient']), uploadSingle('avatar_url'), validate(updateProfileExtendedSchema), patientsController.updateDependentProfile);

// Create patient (Admin, Receptionist)
router.post('/', authorize(['Admin', 'Receptionist']), validate(createPatientSchema), patientsController.createPatient);

// Quick create patient without linked user (Receptionist only)
router.post('/quick', authorize(['Admin', 'Receptionist']), validate(createPatientQuickSchema), patientsController.createPatientQuick);

// Update patient (Admin, Receptionist)
router.put('/:id', authorize(['Admin', 'Receptionist']), validate(updatePatientSchema), patientsController.updatePatient);

// Delete patient (Admin only)
router.delete('/:id', authorize(['Admin']), patientsController.deletePatient);

module.exports = router;