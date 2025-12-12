const Joi = require('joi');

const createSupplierSchema = Joi.object({
  name: Joi.string().max(255).required(),
  contact_info: Joi.object({
    phone: Joi.string().optional(),
    email: Joi.string().email({ tlds: { allow: false } }).optional(),
    address: Joi.string().optional(),
    contact_person: Joi.string().optional(),
    website: Joi.string().uri().optional()
  }).optional().default({})
});

const updateSupplierSchema = Joi.object({
  name: Joi.string().max(255).optional(),
  contact_info: Joi.object({
    phone: Joi.string().optional(),
    email: Joi.string().email({ tlds: { allow: false } }).optional(),
    address: Joi.string().optional(),
    contact_person: Joi.string().optional(),
    website: Joi.string().uri().optional()
  }).optional()
});

module.exports = {
  createSupplierSchema,
  updateSupplierSchema
};