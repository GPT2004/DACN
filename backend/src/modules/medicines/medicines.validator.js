const Joi = require('joi');

const createMedicineSchema = Joi.object({
  name: Joi.string().required(),
  code: Joi.string().optional(),
  description: Joi.string().optional(),
  unit: Joi.string().required(),
  price: Joi.number().integer().min(0).required(),
});

const updateMedicineSchema = Joi.object({
  name: Joi.string().optional(),
  code: Joi.string().optional(),
  description: Joi.string().optional(),
  unit: Joi.string().optional(),
  price: Joi.number().integer().min(0).optional(),
});

const createStockSchema = Joi.object({
  medicine_id: Joi.number().integer().required(),
  batch_number: Joi.string().required(),
  expiry_date: Joi.date().greater('now').required(),
  quantity: Joi.number().integer().min(1).required(),
});

const updateStockSchema = Joi.object({
  batch_number: Joi.string().optional(),
  expiry_date: Joi.date().greater('now').optional(),
  quantity: Joi.number().integer().min(0).optional(),
});

const importMedicinesSchema = Joi.object({
  medicines: Joi.array().items(
    Joi.object({
      name: Joi.string().required(),
      code: Joi.string().optional(),
      description: Joi.string().optional(),
      unit: Joi.string().optional(),
      price: Joi.number().integer().min(0).optional(),
      batch_number: Joi.string().optional(),
      expiry_date: Joi.string().required(), // YYYY-MM-DD format
      quantity: Joi.number().integer().min(1).required(),
    })
  ).min(1).required(),
});

module.exports = {
  createMedicineSchema,
  updateMedicineSchema,
  createStockSchema,
  updateStockSchema,
  importMedicinesSchema,
};