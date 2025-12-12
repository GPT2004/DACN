// src/services/auditLogService.js
import api from './api';

export const auditLogService = {
  // Get all audit logs (Admin)
  getAllAuditLogs: async (params = {}) => {
    return await api.get('/audit_logs', { params });
  },

  // Get audit log by ID (Admin)
  getAuditLogById: async (id) => {
    return await api.get(`/audit_logs/${id}`);
  },

  // Get user audit logs (Admin)
  getUserAuditLogs: async (userId, params = {}) => {
    return await api.get(`/audit_logs/user/${userId}`, { params });
  },

  // Get action types (Admin)
  getActionTypes: async () => {
    return await api.get('/audit_logs/actions/types');
  },

  // Create audit log (Admin)
  createAuditLog: async (action, meta) => {
    return await api.post('/audit_logs', { action, meta });
  },

  // Delete old logs (Admin)
  deleteOldLogs: async (days = 90) => {
    return await api.delete('/audit_logs/cleanup', { params: { days } });
  },
};