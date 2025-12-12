const { errorResponse } = require('../utils/response');
const prisma = require('../config/database');

// Normalize role value to a lowercase string
const normalizeRole = (role) => {
  if (!role) return '';
  if (typeof role === 'string') return role.toLowerCase();
  if (typeof role === 'object' && role.name) return String(role.name).toLowerCase();
  return '';
};

// Map common role aliases to canonical names
const roleAliases = {
  reception: 'receptionist',
};

const mapAlias = (r) => roleAliases[r] || r;

const authorize = (allowedRoles) => {
  return (req, res, next) => {
    if (!req.user) {
      return errorResponse(res, 'Unauthorized', 401);
    }

    // Get user's raw role and normalize it
    const userRoleRaw = req.user.role?.name || req.user.role || '';
    let userRole = String(userRoleRaw).toLowerCase().trim();
    userRole = mapAlias(userRole);

    // Normalize allowed roles to lowercase for comparison
    const roles = (Array.isArray(allowedRoles) ? allowedRoles : [allowedRoles])
      .map((r) => mapAlias(String(r).toLowerCase().trim()));

    // Flexible matching: allow exact match or substring matches to account for
    // slightly different role labels like 'reception account' vs 'receptionist'.
    const matched = roles.some(r => {
      if (!r) return false;
      if (r === userRole) return true;
      if (userRole.includes(r)) return true;
      if (r.includes(userRole)) return true;
      return false;
    });

    if (!matched) {
      // Log helpful debug info for Forbidden attempts
      console.error(`Forbidden request: userId=${req.user?.id} role='${userRoleRaw}' normalized='${userRole}' allowed=[${roles.join(',')}], path=${req.originalUrl}`);
      return errorResponse(res, 'Forbidden: Insufficient permissions', 403);
    }

    next();
  };
};

const authorizeOwner = (resourceType) => {
  return async (req, res, next) => {
    try {
      const resourceId = parseInt(req.params.id);
      const userId = req.user.id;
      const userRoleRaw = req.user.role?.name || req.user.role || '';
      const userRole = mapAlias(String(userRoleRaw).toLowerCase());

      // Admin has full access
      if (userRole === 'admin') {
        return next();
      }

      let resource;

      switch (resourceType) {
        case 'patient': {
          resource = await prisma.patients.findUnique({
            where: { id: resourceId },
          });
          if (resource && resource.user_id !== userId) {
            return errorResponse(res, 'You can only access your own data', 403);
          }
          break;
        }

        case 'appointment': {
          resource = await prisma.appointments.findUnique({
            where: { id: resourceId },
            include: { patient: true, doctor: true },
          });

          const isPatientOwner = resource?.patient?.user_id === userId;
          const isDoctorOwner = resource?.doctor?.user_id === userId;

          if (!isPatientOwner && !isDoctorOwner && userRole !== 'receptionist') {
            return errorResponse(res, 'You can only access your own appointments', 403);
          }
          break;
        }

        case 'medical-record': {
          resource = await prisma.medical_records.findUnique({
            where: { id: resourceId },
            include: { patient: true, doctor: true },
          });

          const isPatient = resource?.patient?.user_id === userId;
          const isDoctor = resource?.doctor?.user_id === userId;

          if (!isPatient && !isDoctor) {
            return errorResponse(res, 'You can only access authorized medical records', 403);
          }
          break;
        }

        default:
          return errorResponse(res, 'Invalid resource type', 400);
      }

      if (!resource) {
        return errorResponse(res, 'Resource not found', 404);
      }

      req.resource = resource;
      next();
    } catch (error) {
      next(error);
    }
  };
};

module.exports = { authorize, authorizeOwner };
