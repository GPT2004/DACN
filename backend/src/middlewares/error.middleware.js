const logger = require('../utils/logger');
const { errorResponse } = require('../utils/response');

const errorHandler = (err, req, res, _next) => {
  logger.error('Error:', {
    message: err.message,
    stack: err.stack,
    url: req.originalUrl,
    method: req.method,
    ip: req.ip,
    user: req.user?.id,
  });

  // Prisma errors
  if (err.code === 'P2002') {
    return errorResponse(res, 'Duplicate entry', 409, {
      field: err.meta?.target,
    });
  }

  if (err.code === 'P2025') {
    return errorResponse(res, 'Record not found', 404);
  }

  if (err.code === 'P2003') {
    return errorResponse(res, 'Foreign key constraint failed', 400);
  }

  // JWT errors
  if (err.name === 'JsonWebTokenError') {
    return errorResponse(res, 'Invalid token', 401);
  }

  if (err.name === 'TokenExpiredError') {
    return errorResponse(res, 'Token expired', 401);
  }

  // Validation errors
  if (err.name === 'ValidationError') {
    return errorResponse(res, err.message, 400);
  }

  // Default error
  return errorResponse(
    res,
    err.message || 'Internal server error',
    err.statusCode || 500
  );
};

const notFoundHandler = (req, res) => {
  return errorResponse(res, `Route ${req.originalUrl} not found`, 404);
};

module.exports = { errorHandler, notFoundHandler };