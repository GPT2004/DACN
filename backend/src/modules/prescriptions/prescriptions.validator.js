const Joi = require('joi');

const prescriptionItemSchema = Joi.object({
  medicine_id: Joi.number().integer().required(),
  quantity: Joi.number().integer().min(1).required(),
  unit_price: Joi.number().integer().min(0).optional(),
  instructions: Joi.string().max(500).optional(),
  dosage: Joi.string().max(200).optional(),
});

const createPrescriptionSchema = Joi.object({
  appointment_id: Joi.number().integer().optional(),
  patient_id: Joi.number().integer().required(),
  doctor_id: Joi.number().integer().optional(), // Optional for doctors
  items: Joi.array().items(prescriptionItemSchema).min(1).required(),
});

const updatePrescriptionSchema = Joi.object({
  items: Joi.array().items(prescriptionItemSchema).min(1).optional(),
});

const dispensePrescriptionSchema = Joi.object({
  items: Joi.array().items(Joi.object({
    medicine_id: Joi.number().integer().required(),
    dispense_quantity: Joi.number().integer().min(0).required(),
  })).min(1).required(),
});

module.exports = {
  createPrescriptionSchema,
  updatePrescriptionSchema,
  dispensePrescriptionSchema,
};