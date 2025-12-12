const Joi = require('joi');

const registerSchema = Joi.object({
  email: Joi.string().email({ tlds: { allow: false } }).required(),
  password: Joi.string().min(6).required(),
  full_name: Joi.string().required(),
  confirm_password: Joi.any().valid(Joi.ref('password')).required().messages({ 'any.only': 'Mật khẩu xác nhận không khớp' }),
  phone: Joi.string().pattern(/^[0-9]{10,11}$/).required(),
  dob: Joi.date().max('now').optional(),
  gender: Joi.string().valid('Male', 'Female', 'Other').optional(),
  blood_type: Joi.string().optional(),
  allergies: Joi.string().optional(),
});

const loginSchema = Joi.object({
  email: Joi.string().email({ tlds: { allow: false } }).required(),
  password: Joi.string().required(),
});

const changePasswordSchema = Joi.object({
  current_password: Joi.string().required(),
  new_password: Joi.string().min(6).required(),
});

module.exports = {
  registerSchema,
  loginSchema,
  changePasswordSchema,
};