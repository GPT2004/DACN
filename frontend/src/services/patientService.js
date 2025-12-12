// src/services/patientService.js
import api from './api';

export const patientService = {
  // Get all patients (Admin/Receptionist/Doctor)
  getPatients: async (params = {}) => {
    return await api.get('/patients', { params });
  },

  getAllPatients: async (params = {}) => {
    return await api.get('/patients', { params });
  },

  // Get patient by ID
  getPatientById: async (id) => {
    return await api.get(`/patients/${id}`);
  },

  // Get my profile (Patient)
  getMyProfile: async () => {
    return await api.get('/patients/me/profile');
  },

  // Get my patients list (own account + dependents)
  getMyPatients: async () => {
    return await api.get('/patients/me/list');
  },

  // Update my profile (Patient)
  updateMyProfile: async (data) => {
    if (typeof FormData !== 'undefined' && data instanceof FormData) {
      return await api.put('/patients/me/profile', data, { headers: { 'Content-Type': 'multipart/form-data' } });
    }
    return await api.put('/patients/me/profile', data);
  },

  // Get patient medical records
  getPatientMedicalRecords: async (patientId, params = {}) => {
    return await api.get(`/patients/${patientId}/medical-records`, { params });
  },

  // Get patient appointments
  getPatientAppointments: async (patientId, params = {}) => {
    return await api.get(`/patients/${patientId}/appointments`, { params });
  },

  // Get patient prescriptions
  getPatientPrescriptions: async (patientId, params = {}) => {
    return await api.get(`/patients/${patientId}/prescriptions`, { params });
  },

  // Get patient invoices
  getPatientInvoices: async (patientId, params = {}) => {
    return await api.get(`/patients/${patientId}/invoices`, { params });
  },

  // Create patient (Receptionist/Admin)
  createPatient: async (data) => {
    return await api.post('/patients', data);
  },

  // Quick create patient without creating linked user (Receptionist)
  createPatientQuick: async (data) => {
    return await api.post('/patients/quick', data);
  },

  // Create dependent patient profile (Patient)
  createDependentPatient: async (data) => {
    return await api.post('/patients/me', data);
  },

  // Update patient (Receptionist/Admin)
  updatePatient: async (id, data) => {
    return await api.put(`/patients/${id}`, data);
  },

  // Update dependent patient as the logged-in patient owner
  updateDependentPatient: async (id, data) => {
    return await api.put(`/patients/me/${id}`, data);
  },

  // Delete patient (Admin)
  deletePatient: async (id) => {
    return await api.delete(`/patients/${id}`);
  },
  // Delete dependent patient as the logged-in patient owner
  deleteDependentPatient: async (id) => {
    return await api.delete(`/patients/me/${id}`);
  },

  // Delete logged-in patient's own profile
  deleteMyProfile: async () => {
    return await api.delete('/patients/me/profile');
  },
};