const Joi = require('joi');

const createRoomSchema = Joi.object({
  name: Joi.string().max(150).required(),
  type: Joi.string().max(100).required(),
  description: Joi.string().optional().allow(''),
  capacity: Joi.number().integer().min(1).default(1)
});

const updateRoomSchema = Joi.object({
  name: Joi.string().max(150).optional(),
  type: Joi.string().max(100).optional(),
  description: Joi.string().optional().allow(''),
  capacity: Joi.number().integer().min(1).optional()
});

const updateRoomStatusSchema = Joi.object({
  status: Joi.string().valid('available', 'occupied', 'maintenance', 'unavailable').required()
});

module.exports = {
  createRoomSchema,
  updateRoomSchema,
  updateRoomStatusSchema
};