const Joi = require('joi');

const createTimeslotSchema = Joi.object({
  doctor_id: Joi.number().integer().positive().required(),
  date: Joi.date().min('now').required(),
  start_time: Joi.string().pattern(/^([0-1][0-9]|2[0-3]):[0-5][0-9]$/).required(),
  end_time: Joi.string().pattern(/^([0-1][0-9]|2[0-3]):[0-5][0-9]$/).required(),
  max_patients: Joi.number().integer().min(1).default(1)
});

const createMultipleTimeslotsSchema = Joi.object({
  doctor_id: Joi.number().integer().positive().required(),
  date: Joi.date().min('now').required(),
  start_time: Joi.string().pattern(/^([0-1][0-9]|2[0-3]):[0-5][0-9]$/).required(),
  end_time: Joi.string().pattern(/^([0-1][0-9]|2[0-3]):[0-5][0-9]$/).required(),
  duration: Joi.number().integer().min(5).max(120).default(20),
  max_patients: Joi.number().integer().min(1).default(1)
});

const updateTimeslotSchema = Joi.object({
  date: Joi.date().min('now').optional(),
  start_time: Joi.string().pattern(/^([0-1][0-9]|2[0-3]):[0-5][0-9]$/).optional(),
  end_time: Joi.string().pattern(/^([0-1][0-9]|2[0-3]):[0-5][0-9]$/).optional(),
  max_patients: Joi.number().integer().min(1).optional(),
  is_active: Joi.boolean().optional()
});

const toggleTimeslotStatusSchema = Joi.object({
  is_active: Joi.boolean().required()
});

const bulkUpdateTimeslotsSchema = Joi.object({
  doctor_id: Joi.number().integer().positive().required(),
  date: Joi.date().required(),
  is_active: Joi.boolean().required()
});

module.exports = {
  createTimeslotSchema,
  createMultipleTimeslotsSchema,
  updateTimeslotSchema,
  toggleTimeslotStatusSchema,
  bulkUpdateTimeslotsSchema
};