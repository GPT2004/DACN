const { verifyToken } = require('../utils/jwt');
const prisma = require('../config/database');
const { errorResponse } = require('../utils/response');

const authenticate = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return errorResponse(res, 'No token provided', 401);
    }

    const token = authHeader.split(' ')[1];
    const decoded = verifyToken(token);

    const user = await prisma.users.findUnique({
      where: { id: decoded.userId },
      include: {
        role: true,
        patients: true,
        doctors: true,
      },
    });

    if (!user || !user.is_active) {
      return errorResponse(res, 'User not found or inactive', 401);
    }

    req.user = user;
    next();
  } catch (error) {
    return errorResponse(res, 'Invalid or expired token', 401);
  }
};

const optionalAuth = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;

    if (authHeader && authHeader.startsWith('Bearer ')) {
      const token = authHeader.split(' ')[1];
      const decoded = verifyToken(token);

      const user = await prisma.users.findUnique({
        where: { id: decoded.userId },
        include: {
          role: true,
          patients: true,
          doctors: true,
        },
      });

      if (user && user.is_active) {
        req.user = user;
      }
    }

    next();
  } catch (error) {
    next();
  }
};

module.exports = { authenticate, optionalAuth };