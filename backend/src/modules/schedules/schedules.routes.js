const express = require('express');
const router = express.Router();
const scheduleController = require('./schedules.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const {
  createScheduleSchema,
  updateScheduleSchema,
  createTimeslotSchema,
  updateTimeslotSchema,
} = require('./schedules.validator');
const { ROLES } = require('../../config/constants');

// Schedule routes
router.get(
  '/',
  authenticate,
  scheduleController.getSchedules
);

// Timeslot routes - MUST come before /:id route!
router.get(
  '/timeslots/available',
  scheduleController.getAvailableTimeslots
);

router.get(
  '/:id',
  authenticate,
  scheduleController.getScheduleById
);

router.post(
  '/',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN, ROLES.RECEPTIONIST]),
  validate(createScheduleSchema),
  scheduleController.createSchedule
);

router.put(
  '/:id',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  validate(updateScheduleSchema),
  scheduleController.updateSchedule
);

router.delete(
  '/:id',
  authenticate,
  authorize([ROLES.DOCTOR, ROLES.ADMIN]),
  scheduleController.deleteSchedule
);

// Timeslot routes - continued
router.post(
  '/timeslots',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.RECEPTIONIST]),
  validate(createTimeslotSchema),
  scheduleController.createTimeslot
);

router.put(
  '/timeslots/:id',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.RECEPTIONIST]),
  validate(updateTimeslotSchema),
  scheduleController.updateTimeslot
);

router.delete(
  '/timeslots/:id',
  authenticate,
  authorize([ROLES.ADMIN, ROLES.RECEPTIONIST]),
  scheduleController.deleteTimeslot
);

module.exports = router;