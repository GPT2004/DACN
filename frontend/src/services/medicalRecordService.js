// src/services/medicalRecordService.js
import api from './api';

export const medicalRecordService = {
  // Get medical records
  getMedicalRecords: async (params = {}) => {
    return await api.get('/medical-records', { params });
  },

  // Get medical record by ID
  getMedicalRecordById: async (id) => {
    return await api.get(`/medical-records/${id}`);
  },

  // Get patient medical history
  getPatientMedicalHistory: async (patientId) => {
    return await api.get(`/medical-records/patient/${patientId}/history`);
  },

  // Create medical record (Doctor)
  createMedicalRecord: async (data) => {
    return await api.post('/medical-records', data);
  },

  // Update medical record (Doctor)
  updateMedicalRecord: async (id, data) => {
    return await api.put(`/medical-records/${id}`, data);
  },

  // Delete medical record (Admin)
  deleteMedicalRecord: async (id) => {
    return await api.delete(`/medical-records/${id}`);
  },

  // Send medical record to patient (Doctor)
  sendToPatient: async (id) => {
    return await api.post(`/medical-records/${id}/send-to-patient`);
  },

  // Upload attachment
  uploadAttachment: async (id, file) => {
    const formData = new FormData();
    formData.append('file', file);
    return await api.post(`/medical-records/${id}/attachments`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
};