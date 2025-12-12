const Joi = require('joi');

const createMedicalRecordSchema = Joi.object({
  appointment_id: Joi.number().integer().optional(),
  patient_id: Joi.number().integer().required(),
  doctor_id: Joi.number().integer().optional(), // Optional for doctors
  diagnosis: Joi.string().required(),
  notes: Joi.string().optional().allow(null, ''),
  exam_results: Joi.string().optional().allow(null, ''),
  lab_tests: Joi.string().optional().allow(null, ''),
  attachments: Joi.array().items(
    Joi.object({
      filename: Joi.string().required(),
      path: Joi.string().required(),
      mimetype: Joi.string().optional(),
      size: Joi.number().optional(),
    })
  ).optional().allow(null),
});

const updateMedicalRecordSchema = Joi.object({
  diagnosis: Joi.string().optional(),
  notes: Joi.string().optional().allow(null, ''),
  exam_results: Joi.string().optional().allow(null, ''),
  lab_tests: Joi.string().optional().allow(null, ''),
  attachments: Joi.array().items(
    Joi.object({
      filename: Joi.string().required(),
      path: Joi.string().required(),
      mimetype: Joi.string().optional(),
      size: Joi.number().optional(),
    })
  ).optional().allow(null),
});

module.exports = {
  createMedicalRecordSchema,
  updateMedicalRecordSchema,
};