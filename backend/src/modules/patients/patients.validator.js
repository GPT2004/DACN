const Joi = require('joi');

const createPatientSchema = Joi.object({
  email: Joi.string().email({ tlds: { allow: false } }).required(),
  password: Joi.string().min(6).required(),
  full_name: Joi.string().required(),
  phone: Joi.string().pattern(/^[0-9]{10,11}$/).required(),
  dob: Joi.date().max('now').optional(),
  gender: Joi.string().valid('Male', 'Female', 'Other').optional(),
  blood_type: Joi.string().valid('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-').optional(),
  allergies: Joi.string().max(500).optional().allow(''),
  emergency_contact: Joi.object({
    name: Joi.string().optional(),
    phone: Joi.string().optional(),
    relationship: Joi.string().optional()
  }).optional()
});

// Quick create for receptionist: do not create linked user account
const createPatientQuickSchema = Joi.object({
  full_name: Joi.string().required(),
  phone: Joi.string().pattern(/^[0-9]{9,11}$/).required(),
  dob: Joi.date().max('now').optional(),
  gender: Joi.string().valid('Male', 'Female', 'Other', 'MALE', 'FEMALE', 'OTHER').optional(),
  blood_type: Joi.string().valid('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-').optional(),
  allergies: Joi.string().max(500).optional().allow(''),
  email: Joi.string().email({ tlds: { allow: false } }).optional().allow('', null),
  address: Joi.string().max(500).optional().allow(''),
  emergency_contact: Joi.object({
    name: Joi.string().optional(),
    phone: Joi.string().optional(),
    relationship: Joi.string().optional()
  }).optional()
});

const updatePatientSchema = Joi.object({
  full_name: Joi.string().optional(),
  phone: Joi.string().pattern(/^[0-9]{10,11}$/).optional(),
  email: Joi.string().email({ tlds: { allow: false } }).optional(),
  dob: Joi.date().max('now').optional(),
  gender: Joi.string().valid('Male', 'Female', 'Other').optional(),
  blood_type: Joi.string().valid('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-').optional(),
  allergies: Joi.string().max(500).optional().allow(''),
  emergency_contact: Joi.object({
    name: Joi.string().optional(),
    phone: Joi.string().optional(),
    relationship: Joi.string().optional()
  }).optional()
});

const updateProfileSchema = Joi.object({
  full_name: Joi.string().optional(),
  phone: Joi.string().pattern(/^[0-9]{9,11}$/).optional().allow(''),
  dob: Joi.date().max('now').optional().allow(''),
  gender: Joi.string().valid('Male', 'Female', 'Other', 'MALE', 'FEMALE', 'OTHER').optional().allow(''),
  address: Joi.string().max(500).optional().allow(''),
  avatar_url: Joi.string().optional().allow('', null),
  emergency_contact: Joi.object({
    name: Joi.string().optional(),
    phone: Joi.string().optional(),
    relationship: Joi.string().optional()
  }).optional()
});

// Extended profile fields to match front-end form
const extendedProfileFields = Joi.object({
  occupation: Joi.string().valid('STUDENT','EMPLOYEE','SELF_EMPLOYED','RETIRED','UNEMPLOYED','OTHER').optional().allow(''),
  id_type: Joi.string().valid('CCCD','CMND','PASSPORT').optional().allow(''),
  id_number: Joi.string().max(100).optional().allow(''),
  nationality: Joi.string().max(100).optional().allow(''),
  ethnicity: Joi.string().max(100).optional().allow(''),
  address: Joi.string().max(500).optional().allow(''),
});

const updateProfileExtendedSchema = updateProfileSchema.concat(extendedProfileFields);

const createDependentPatientSchema = Joi.object({
  full_name: Joi.string().required(),
  phone: Joi.string().pattern(/^[0-9]{9,11}$/).required(),
  dob: Joi.date().max('now').optional(),
  gender: Joi.string().valid('Male', 'Female', 'Other', 'MALE', 'FEMALE', 'OTHER').optional(),
  blood_type: Joi.string().valid('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-').optional(),
  allergies: Joi.string().max(500).optional().allow(''),
  email: Joi.string().email({ tlds: { allow: false } }).optional().allow('', null),
  address: Joi.string().max(500).optional().allow(''),
  emergency_contact: Joi.object({
    name: Joi.string().optional(),
    phone: Joi.string().optional(),
    relationship: Joi.string().optional()
  }).optional()
});

// Extend createDependentPatient schema with additional identity/contact fields
const createDependentPatientExtendedSchema = createDependentPatientSchema.keys({
  occupation: Joi.string().max(255).optional().allow(''),
  id_type: Joi.string().max(50).optional().allow(''),
  id_number: Joi.string().max(100).optional().allow(''),
  id_issue_date: Joi.date().optional().allow(null),
  id_issue_place: Joi.string().max(255).optional().allow(''),
  nationality: Joi.string().max(100).optional().allow(''),
  ethnicity: Joi.string().max(100).optional().allow(''),
  zalo: Joi.boolean().optional()
});

module.exports = {
  createPatientSchema,
  updatePatientSchema,
  updateProfileSchema,
  createDependentPatientSchema,
  createDependentPatientExtendedSchema,
  createPatientQuickSchema,
  updateProfileExtendedSchema
};