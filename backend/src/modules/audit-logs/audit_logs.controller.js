const auditLogsService = require('./audit-logs.service');
const { successResponse } = require('../../utils/response');

class AuditLogsController {
  async getAllAuditLogs(req, res, next) {
    try {
      const { 
        page = 1, 
        limit = 20, 
        user_id, 
        action, 
        startDate, 
        endDate 
      } = req.query;

      const filters = {};
      if (user_id) filters.user_id = parseInt(user_id);
      if (action) filters.action = action;
      if (startDate) filters.startDate = startDate;
      if (endDate) filters.endDate = endDate;

      const result = await auditLogsService.getAllAuditLogs(filters, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Audit logs retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getAuditLogById(req, res, next) {
    try {
      const { id } = req.params;
      const log = await auditLogsService.getAuditLogById(parseInt(id));

      if (!log) {
        throw new Error('Audit log not found');
      }

      return successResponse(res, log, 'Audit log retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async createAuditLog(req, res, next) {
    try {
      const logData = req.validatedBody;
      const userId = req.user.id;

      const newLog = await auditLogsService.createAuditLog({
        ...logData,
        user_id: userId
      });

      return successResponse(res, newLog, 'Audit log created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async getUserAuditLogs(req, res, next) {
    try {
      const { userId } = req.params;
      const { page = 1, limit = 20, action, startDate, endDate } = req.query;

      const filters = {};
      if (action) filters.action = action;
      if (startDate) filters.startDate = startDate;
      if (endDate) filters.endDate = endDate;

      const result = await auditLogsService.getUserAuditLogs(
        parseInt(userId),
        filters,
        { page: parseInt(page), limit: parseInt(limit) }
      );

      return successResponse(res, result, 'User audit logs retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getActionTypes(req, res, next) {
    try {
      const actionTypes = await auditLogsService.getActionTypes();
      return successResponse(res, actionTypes, 'Action types retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteOldLogs(req, res, next) {
    try {
      const { days = 90 } = req.query;
      const result = await auditLogsService.deleteOldLogs(parseInt(days));

      return successResponse(res, result, 'Old audit logs deleted successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new AuditLogsController();