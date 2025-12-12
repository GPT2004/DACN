const express = require('express');
const router = express.Router();
const timeslotsController = require('./timeslots.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const { 
  createTimeslotSchema, 
  updateTimeslotSchema,
  createMultipleTimeslotsSchema,
  toggleTimeslotStatusSchema,
  bulkUpdateTimeslotsSchema
} = require('./timeslots.validator');

router.use(authenticate);

// Get all timeslots
router.get('/', 
  authorize(['Admin', 'Receptionist', 'Doctor', 'Patient']), 
  timeslotsController.getAllTimeslots
);

// Get available timeslots (must be before /:id routes)
router.get('/available', 
  authorize(['Admin', 'Receptionist', 'Doctor', 'Patient']), 
  timeslotsController.getAvailableTimeslots
);

// Get doctor timeslots
router.get('/doctor/:doctorId', 
  authorize(['Admin', 'Receptionist', 'Doctor', 'Patient']), 
  timeslotsController.getDoctorTimeslots
);

// Get timeslot by ID
router.get('/:id', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  timeslotsController.getTimeslotById
);

// Get timeslot appointments
router.get('/:id/appointments', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  timeslotsController.getTimeslotAppointments
);

// Create single timeslot
router.post('/', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  validate(createTimeslotSchema), 
  timeslotsController.createTimeslot
);

// Create multiple timeslots (bulk)
router.post('/bulk', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  validate(createMultipleTimeslotsSchema), 
  timeslotsController.createMultipleTimeslots
);

// Update timeslot
router.put('/:id', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  validate(updateTimeslotSchema), 
  timeslotsController.updateTimeslot
);

// Toggle timeslot status
router.patch('/:id/status', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  validate(toggleTimeslotStatusSchema), 
  timeslotsController.toggleTimeslotStatus
);

// Bulk update timeslots
router.patch('/bulk/update', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  validate(bulkUpdateTimeslotsSchema), 
  timeslotsController.bulkUpdateTimeslots
);

// Delete timeslot
router.delete('/:id', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  timeslotsController.deleteTimeslot
);

module.exports = router;
