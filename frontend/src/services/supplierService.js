// src/services/supplierService.js
import api from './api';

export const supplierService = {
  // Get all suppliers
  getAllSuppliers: async (params = {}) => {
    return await api.get('/suppliers', { params });
  },

  // Get supplier by ID
  getSupplierById: async (id) => {
    return await api.get(`/suppliers/${id}`);
  },

  // Get supplier stats
  getSupplierStats: async (id) => {
    return await api.get(`/suppliers/${id}/stats`);
  },

  // Create supplier (Admin)
  createSupplier: async (data) => {
    return await api.post('/suppliers', data);
  },

  // Update supplier (Admin)
  updateSupplier: async (id, data) => {
    return await api.put(`/suppliers/${id}`, data);
  },

  // Delete supplier (Admin)
  deleteSupplier: async (id) => {
    return await api.delete(`/suppliers/${id}`);
  },
};