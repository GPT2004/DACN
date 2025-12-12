const Joi = require('joi');

const createUserSchema = Joi.object({
  email: Joi.string().email({ tlds: { allow: false } }).required(),
  password: Joi.string().min(6).required(),
  full_name: Joi.string().required(),
  phone: Joi.string().pattern(/^[0-9]{10,11}$/).required(),
  dob: Joi.date().max('now').optional(),
  role_id: Joi.number().integer().positive().required(),
  avatar_url: Joi.string().uri().optional().allow('')
});

const updateUserSchema = Joi.object({
  email: Joi.string().email({ tlds: { allow: false } }).optional(),
  full_name: Joi.string().optional(),
  phone: Joi.string().pattern(/^[0-9]{10,11}$/).optional(),
  dob: Joi.date().max('now').optional(),
  avatar_url: Joi.string().uri().optional().allow('')
});

const updateProfileSchema = Joi.object({
  full_name: Joi.string().optional(),
  phone: Joi.string().pattern(/^[0-9]{10,11}$/).optional(),
  avatar_url: Joi.string().uri().optional().allow('')
});

const changePasswordSchema = Joi.object({
  current_password: Joi.string().required(),
  new_password: Joi.string().min(6).required()
});

const resetPasswordSchema = Joi.object({
  new_password: Joi.string().min(6).required()
});

const toggleUserStatusSchema = Joi.object({
  is_active: Joi.boolean().required()
});

const assignRoleSchema = Joi.object({
  role_id: Joi.number().integer().positive().required()
});

module.exports = {
  createUserSchema,
  updateUserSchema,
  updateProfileSchema,
  changePasswordSchema,
  resetPasswordSchema,
  toggleUserStatusSchema,
  assignRoleSchema
};