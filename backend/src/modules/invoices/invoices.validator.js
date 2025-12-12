const Joi = require('joi');

const invoiceItemSchema = Joi.object({
  type: Joi.string().valid('consultation', 'medicine', 'lab', 'other').required(),
  description: Joi.string().required(),
  quantity: Joi.number().integer().min(1).required(),
  unit_price: Joi.number().integer().min(0).required(),
});

const createInvoiceSchema = Joi.object({
  appointment_id: Joi.number().integer().optional(),
  patient_id: Joi.number().integer().required(),
  items: Joi.array().items(invoiceItemSchema).min(1).required(),
  tax: Joi.number().integer().min(0).optional(),
  discount: Joi.number().integer().min(0).optional(),
});

const updateInvoiceSchema = Joi.object({
  items: Joi.array().items(invoiceItemSchema).min(1).optional(),
  discount: Joi.number().integer().min(0).optional(),
});

const payInvoiceSchema = Joi.object({
  payment_method: Joi.string().valid('cash', 'card', 'transfer', 'insurance').required(),
  paid_amount: Joi.number().integer().min(0).optional(),
});

const refundInvoiceSchema = Joi.object({
  reason: Joi.string().max(500).required(),
});

module.exports = {
  createInvoiceSchema,
  updateInvoiceSchema,
  payInvoiceSchema,
  refundInvoiceSchema,
};