const Joi = require('joi');

const createAppointmentSchema = Joi.object({
  patient_id: Joi.number().integer().optional(), // Optional for patients (auto-filled)
  timeslot_id: Joi.number().integer().required(),
  appointment_date: Joi.date().optional(), // Optional, will use timeslot date
  reason: Joi.string().max(500).required(),
});

const updateAppointmentSchema = Joi.object({
  appointment_date: Joi.date().optional(),
  appointment_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).optional(),
  reason: Joi.string().max(500).optional(),
  status: Joi.string()
    .valid('PENDING', 'CONFIRMED', 'CHECKED_IN', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED', 'NO_SHOW')
    .optional(),
});

const cancelAppointmentSchema = Joi.object({
  reason: Joi.string().max(500).optional(),
});

const rescheduleAppointmentSchema = Joi.object({
  timeslotId: Joi.number().integer().required(),
  reason: Joi.string().max(500).optional()
});

module.exports = {
  createAppointmentSchema,
  updateAppointmentSchema,
  cancelAppointmentSchema,
  rescheduleAppointmentSchema
};