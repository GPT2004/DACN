const Joi = require('joi');

const createNotificationSchema = Joi.object({
  user_id: Joi.number().integer().positive().optional(),
  role_name: Joi.string().valid('Admin', 'Doctor', 'Patient', 'Receptionist', 'Pharmacist', 'LabTech').optional(),
  type: Joi.string().max(100).required(),
  payload: Joi.object().optional()
}).or('user_id', 'role_name');

module.exports = {
  createNotificationSchema
};