// src/services/prescriptionService.js
import api from './api';

export const prescriptionService = {
  // Get prescriptions
  getPrescriptions: async (params = {}) => {
    return await api.get('/prescriptions', { params });
  },

  // Get prescription by ID
  getPrescriptionById: async (id) => {
    return await api.get(`/prescriptions/${id}`);
  },

  // Create prescription (Doctor)
  createPrescription: async (data) => {
    return await api.post('/prescriptions', data);
  },

  // Notify reception that a prescription is ready for invoicing
  notifyReception: async (id) => {
    return await api.post(`/prescriptions/${id}/notify-reception`);
  },
  getForInvoicing: async (params = {}) => {
    return await api.get('/prescriptions/for-invoicing', { params });
  },

  // Update prescription (Doctor, DRAFT only)
  updatePrescription: async (id, data) => {
    return await api.put(`/prescriptions/${id}`, data);
  },

  // Approve prescription (Doctor)
  approvePrescription: async (id) => {
    return await api.patch(`/prescriptions/${id}/approve`);
  },

  // Dispense prescription (Pharmacist)
  dispensePrescription: async (id, data = {}) => {
    return await api.patch(`/prescriptions/${id}/dispense`, data);
  },

  // Delete prescription (Doctor/Admin, DRAFT only)
  deletePrescription: async (id) => {
    return await api.delete(`/prescriptions/${id}`);
  },

  // Download prescription PDF
  downloadPrescriptionPDF: async (id) => {
    return await api.get(`/prescriptions/${id}/pdf`, {
      responseType: 'blob',
    });
  },
};