const Joi = require('joi');

const createScheduleSchema = Joi.object({
  doctor_id: Joi.number().integer().optional(), // Optional for doctors
  room_id: Joi.number().integer().optional(),
  date: Joi.date().min('now').required(),
  start_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).required(),
  end_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).required(),
  recurrent_rule: Joi.string().valid('DAILY', 'WEEKLY', 'MONTHLY').optional(),
  auto_create_timeslots: Joi.boolean().optional(),
  slot_duration: Joi.number().integer().min(10).max(120).optional(),
  slot_duration_minutes: Joi.number().integer().min(10).max(120).optional(), // Frontend sends this
  capacity: Joi.number().integer().optional(), // Frontend might send this too
});

const updateScheduleSchema = Joi.object({
  room_id: Joi.number().integer().optional().allow(null),
  date: Joi.date().optional(),
  start_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).optional(),
  end_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).optional(),
  recurrent_rule: Joi.string().valid('DAILY', 'WEEKLY', 'MONTHLY').optional().allow(null),
});

const createTimeslotSchema = Joi.object({
  doctor_id: Joi.number().integer().required(),
  date: Joi.date().min('now').required(),
  start_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).required(),
  end_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).required(),
  max_patients: Joi.number().integer().min(1).max(10).optional(),
});

const updateTimeslotSchema = Joi.object({
  max_patients: Joi.number().integer().min(1).max(10).optional(),
  is_active: Joi.boolean().optional(),
});

module.exports = {
  createScheduleSchema,
  updateScheduleSchema,
  createTimeslotSchema,
  updateTimeslotSchema,
};