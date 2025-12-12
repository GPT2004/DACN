const prisma = require('../config/database');
const logger = require('../utils/logger');

const auditLog = (action) => {
  return async (req, res, next) => {
    const originalSend = res.send;

    res.send = function (data) {
      // Log to database
      prisma.audit_logs
        .create({
          data: {
            user_id: req.user?.id || null,
            action: action,
            meta: {
              method: req.method,
              url: req.originalUrl,
              ip: req.ip,
              userAgent: req.get('user-agent'),
              body: sanitizeBody(req.body),
              params: req.params,
              query: req.query,
              statusCode: res.statusCode,
            },
          },
        })
        .catch((err) => logger.error('Audit log error:', err));

      originalSend.call(this, data);
    };

    next();
  };
};

const sanitizeBody = (body) => {
  if (!body) return null;
  
  const sanitized = { ...body };
  const sensitiveFields = ['password', 'token', 'secret'];
  
  sensitiveFields.forEach((field) => {
    if (sanitized[field]) {
      sanitized[field] = '***REDACTED***';
    }
  });
  
  return sanitized;
};

module.exports = { auditLog };