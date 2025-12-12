const Joi = require('joi');

const createDoctorSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().min(6).required(),
  full_name: Joi.string().required(),
  phone: Joi.string().pattern(/^[0-9]{10,11}$/).required(),
  dob: Joi.date().max('now').optional(),
  license_number: Joi.string().required(),

  // SỬA TẠI ĐÂY: CHO PHÉP CẢ CHUỖI HOẶC MẢNG
  specialties: Joi.alternatives().try(
    Joi.array().items(Joi.string()).min(1),
    Joi.string().min(1)
  ).required(),

  bio: Joi.string().max(1000).optional().allow(''),
  consultation_fee: Joi.number().integer().min(0).optional()
});

const updateDoctorSchema = Joi.object({
  full_name: Joi.string().optional(),
  phone: Joi.string().pattern(/^[0-9]{10,11}$/).optional().allow(''),
  email: Joi.string().email().optional(),
  dob: Joi.date().max('now').optional(),
  license_number: Joi.string().optional(),

  // Support both 'specialty' (singular from FE) and 'specialties' (plural)
  specialty: Joi.alternatives().try(
    Joi.array().items(Joi.string()),
    Joi.string()
  ).optional(),
  
  specialties: Joi.alternatives().try(
    Joi.array().items(Joi.string()),
    Joi.string()
  ).optional(),

  bio: Joi.string().max(1000).optional().allow(''),
  consultation_fee: Joi.number().integer().min(0).optional()
}).unknown(true);

const updateProfileSchema = Joi.object({
  full_name: Joi.string().optional().allow(''),
  phone: Joi.string().pattern(/^[0-9]{9,11}$/).optional().allow(''),
  avatar_url: Joi.string().optional().allow('').allow(null),
  bio: Joi.string().max(1000).optional().allow(''),
  dob: Joi.date().max('now').optional(),
  gender: Joi.string().valid('MALE', 'FEMALE', 'OTHER').optional().allow(''),
  address: Joi.string().optional().allow(''),
  consultation_fee: Joi.number().integer().min(0).optional(),

  // VÀ ĐÂY
  specialties: Joi.alternatives().try(
    Joi.array().items(Joi.string()),
    Joi.string()
  ).optional()
});

const createScheduleSchema = Joi.object({
  // Accept either a Date parsable value or common date strings (YYYY-MM-DD or MM/DD/YYYY)
  date: Joi.alternatives().try(
    Joi.date().min('now'),
    Joi.string().pattern(/^\d{4}-\d{2}-\d{2}$/),
    Joi.string().pattern(/^\d{1,2}\/\d{1,2}\/\d{4}$/)
  ).required(),
  start_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$/).required(),
  end_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$/).required(),
  room_id: Joi.number().integer().optional().allow(null),
  slot_duration_minutes: Joi.number().integer().min(10).max(120).optional(),
  capacity: Joi.number().integer().min(1).max(10).optional()
});

const updateScheduleSchema = Joi.object({
  date: Joi.date().optional(),
  start_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$/).optional(),
  end_time: Joi.string().pattern(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$/).optional(),
  room_id: Joi.number().integer().optional().allow(null),
  recurrent_rule: Joi.string().valid('DAILY', 'WEEKLY', 'MONTHLY').optional().allow(null)
});

const rescheduleAppointmentSchema = Joi.object({
  timeslotId: Joi.number().integer().required()
});

module.exports = {
  createDoctorSchema,
  updateDoctorSchema,
  updateProfileSchema,
  createScheduleSchema,
  updateScheduleSchema,
  rescheduleAppointmentSchema
};