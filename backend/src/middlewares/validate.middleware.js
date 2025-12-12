const { errorResponse } = require('../utils/response');

const validate = (schema) => {
  return (req, res, next) => {
    // If a file was uploaded via multer, remove file fields from req.body before validation
    // so Joi (which expects strings) does not receive file objects.
    const bodyToValidate = { ...(req.body || {}) };
    try {
      if (req.file && req.file.fieldname) {
        delete bodyToValidate[req.file.fieldname];
      }
      if (req.files) {
        if (Array.isArray(req.files)) {
          req.files.forEach(f => { if (f && f.fieldname) delete bodyToValidate[f.fieldname]; });
        } else if (typeof req.files === 'object') {
          // multer.fields() produces an object where keys are fieldnames
          Object.keys(req.files).forEach(k => { delete bodyToValidate[k]; });
        }
      }
    } catch (e) {
      // ignore errors during cleanup
    }

    const { error, value } = schema.validate(bodyToValidate, {
      abortEarly: false,
      stripUnknown: true,
    });

    if (error) {
      const errors = error.details.map((detail) => ({
        field: detail.path.join('.'),
        message: detail.message,
      }));

      return errorResponse(res, 'Validation error', 400, errors);
    }

    req.validatedBody = value;
    next();
  };
};

const validateQuery = (schema) => {
  return (req, res, next) => {
    const { error, value } = schema.validate(req.query, {
      abortEarly: false,
      stripUnknown: true,
    });

    if (error) {
      const errors = error.details.map((detail) => ({
        field: detail.path.join('.'),
        message: detail.message,
      }));

      return errorResponse(res, 'Query validation error', 400, errors);
    }

    req.validatedQuery = value;
    next();
  };
};

module.exports = { validate, validateQuery };