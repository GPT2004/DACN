const Joi = require('joi');

const createAuditLogSchema = Joi.object({
  action: Joi.string().max(255).required(),
  meta: Joi.object().optional().default({})
});

module.exports = {
  createAuditLogSchema
};