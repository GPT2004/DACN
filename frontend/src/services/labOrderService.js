// src/services/labOrderService.js
import api from './api';

export const labOrderService = {
  // Get lab orders
  getLabOrders: async (params = {}) => {
    return await api.get('/lab-orders', { params });
  },

  // Get lab order by ID
  getLabOrderById: async (id) => {
    return await api.get(`/lab-orders/${id}`);
  },

  // Create lab order (Doctor)
  createLabOrder: async (data) => {
    return await api.post('/lab-orders', data);
  },

  // Update lab order (Doctor, PENDING only)
  updateLabOrder: async (id, data) => {
    return await api.put(`/lab-orders/${id}`, data);
  },

  // Update lab results (Lab Tech)
  updateLabResults: async (id, results) => {
    return await api.patch(`/lab-orders/${id}/results`, { results });
  },

  // Complete lab order (Lab Tech)
  completeLabOrder: async (id) => {
    return await api.patch(`/lab-orders/${id}/complete`);
  },

  // Delete lab order (Doctor/Admin, PENDING only)
  deleteLabOrder: async (id) => {
    return await api.delete(`/lab-orders/${id}`);
  },
};