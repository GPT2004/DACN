const Joi = require('joi');

const reportDateRangeSchema = Joi.object({
  startDate: Joi.date().required(),
  endDate: Joi.date().min(Joi.ref('startDate')).required(),
  groupBy: Joi.string().valid('day', 'week', 'month').optional()
});

const exportReportSchema = Joi.object({
  type: Joi.string().valid('revenue', 'appointments', 'doctors', 'stock').required(),
  format: Joi.string().valid('csv', 'json').default('csv'),
  startDate: Joi.date().optional(),
  endDate: Joi.date().optional()
});

module.exports = {
  reportDateRangeSchema,
  exportReportSchema
};