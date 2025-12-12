const express = require('express');
const router = express.Router();
const DoctorsController = require('./doctors.controller');
const doctorsController = new DoctorsController();
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const { createDoctorSchema, updateDoctorSchema, updateProfileSchema, createScheduleSchema, updateScheduleSchema, rescheduleAppointmentSchema } = require('./doctors.validator');
const { uploadSingle } = require('../../middlewares/upload.middleware');

// Public routes
router.get('/public/specialties/list', doctorsController.getSpecialties);
router.get('/public', doctorsController.getAllDoctorsPublic);
router.get('/public/:id', doctorsController.getDoctorByIdPublic);

// Protected routes
router.use(authenticate);

router.get('/user/:userId', authorize(['Admin', 'Receptionist', 'Doctor']), doctorsController.getDoctorByUserId);
router.get('/me/appointments/today', authorize(['Doctor']), doctorsController.getMyTodayAppointments);
router.get('/', authorize(['Admin', 'Receptionist', 'Doctor']), doctorsController.getAllDoctors);
router.get('/:id', authorize(['Admin', 'Receptionist', 'Doctor']), doctorsController.getDoctorById);
router.get('/:id/appointments', authorize(['Admin', 'Receptionist', 'Doctor']), doctorsController.getDoctorAppointments);
router.post('/:id/schedules', authorize(['Admin', 'Doctor', 'Receptionist']), validate(createScheduleSchema), doctorsController.createSchedule);
router.put('/:id/schedules/:scheduleId', authorize(['Admin', 'Doctor', 'Receptionist']), validate(updateScheduleSchema), doctorsController.updateSchedule);
router.delete('/:id/schedules/:scheduleId', authorize(['Admin', 'Doctor', 'Receptionist']), doctorsController.deleteSchedule);
router.get('/:id/schedules', authorize(['Admin', 'Receptionist', 'Doctor']), doctorsController.getDoctorSchedules);
router.get('/:id/patients', authorize(['Admin', 'Doctor']), doctorsController.getDoctorPatients);
router.get('/:id/stats', authorize(['Admin', 'Doctor']), doctorsController.getDoctorStats);

router.post('/', authorize(['Admin']), validate(createDoctorSchema), doctorsController.createDoctor);
router.put('/:id', authorize(['Admin']), validate(updateDoctorSchema), uploadSingle('avatar_url'), doctorsController.updateDoctor);
router.put('/me/profile', authorize(['Doctor']), uploadSingle('avatar_url'), validate(updateProfileSchema), doctorsController.updateOwnProfile);
router.delete('/:id', authorize(['Admin']), doctorsController.deleteDoctor);
router.patch('/:id/toggle-status', authorize(['Admin']), doctorsController.toggleDoctorStatus);

module.exports = router;