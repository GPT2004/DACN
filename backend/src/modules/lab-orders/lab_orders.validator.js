const Joi = require('joi');

const testSchema = Joi.object({
  test_name: Joi.string().required(),
  test_code: Joi.string().required(),
  description: Joi.string().optional(),
});

const resultSchema = Joi.object({
  test_name: Joi.string().required(),
  test_code: Joi.string().required(),
  value: Joi.string().required(),
  unit: Joi.string().optional(),
  reference_range: Joi.string().optional(),
  status: Joi.string().valid('normal', 'abnormal', 'critical').optional(),
  notes: Joi.string().optional(),
});

const createLabOrderSchema = Joi.object({
  appointment_id: Joi.number().integer().optional(),
  patient_id: Joi.number().integer().required(),
  doctor_id: Joi.number().integer().optional(), // Optional for doctors
  tests: Joi.array().items(testSchema).min(1).required(),
});

const updateLabOrderSchema = Joi.object({
  tests: Joi.array().items(testSchema).min(1).optional(),
  status: Joi.string().valid('PENDING', 'IN_PROGRESS').optional(),
});

const updateLabResultsSchema = Joi.object({
  results: Joi.array().items(resultSchema).min(1).required(),
});

module.exports = {
  createLabOrderSchema,
  updateLabOrderSchema,
  updateLabResultsSchema,
};