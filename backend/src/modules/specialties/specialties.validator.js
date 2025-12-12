const Joi = require('joi');

const createSpecialtySchema = Joi.object({
  name: Joi.string().min(1).max(255).required(),
  slug: Joi.string().min(1).max(255).optional(),
  description: Joi.string().optional(),
  image_url: Joi.string().uri().optional()
});

const updateSpecialtySchema = Joi.object({
  name: Joi.string().min(1).max(255).optional(),
  slug: Joi.string().min(1).max(255).optional(),
  description: Joi.string().optional(),
  image_url: Joi.string().uri().optional()
});

module.exports = {
  createSpecialtySchema,
  updateSpecialtySchema
};