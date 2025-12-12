const Joi = require('joi');

const createStockSchema = Joi.object({
  medicine_id: Joi.number().integer().positive().required(),
  batch_number: Joi.string().max(100).required(),
  expiry_date: Joi.date().greater('now').required(),
  quantity: Joi.number().integer().min(0).required()
});

const updateStockSchema = Joi.object({
  batch_number: Joi.string().max(100).optional(),
  expiry_date: Joi.date().greater('now').optional(),
  quantity: Joi.number().integer().min(0).optional()
});

const adjustStockSchema = Joi.object({
  quantity: Joi.number().integer().not(0).required(),
  reason: Joi.string().max(500).required()
});

module.exports = {
  createStockSchema,
  updateStockSchema,
  adjustStockSchema
};