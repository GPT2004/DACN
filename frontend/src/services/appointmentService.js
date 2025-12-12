// src/services/appointmentService.js
import api from './api';

export const appointmentService = {
  // Get all appointments
  getAppointments: async (params = {}) => {
    return await api.get('/appointments', { params });
  },

  // Get appointment by ID
  getAppointmentById: async (id) => {
    return await api.get(`/appointments/${id}`);
  },

  // Create appointment
  createAppointment: async (data) => {
    return await api.post('/appointments', data);
  },

  // Update appointment
  updateAppointment: async (id, data) => {
    return await api.put(`/appointments/${id}`, data);
  },

  // Confirm appointment (Receptionist/Admin)
  confirmAppointment: async (id) => {
    return await api.patch(`/appointments/${id}/confirm`);
  },

  // Check-in appointment (Receptionist/Admin)
  checkInAppointment: async (id) => {
    return await api.patch(`/appointments/${id}/check-in`);
  },

  // Start appointment (Doctor)
  startAppointment: async (id) => {
    return await api.patch(`/appointments/${id}/start`);
  },

  // Complete appointment (Doctor)
  completeAppointment: async (id) => {
    return await api.patch(`/appointments/${id}/complete`);
  },

  // Mark as no-show (Receptionist/Admin)
  markNoShow: async (id) => {
    return await api.patch(`/appointments/${id}/no-show`);
  },

  // Cancel appointment
  cancelAppointment: async (id, reason) => {
    return await api.patch(`/appointments/${id}/cancel`, { reason });
  },
  // Delete appointment (permanent)
  deleteAppointment: async (id) => {
    return await api.delete(`/appointments/${id}`);
  },

  // Get all appointments (no pagination, no status filter)
  getAllAppointmentsByPatient: async () => {
    return await api.get('/appointments/all/all-appointments');
  },

  // Get appointment history (past appointments for patient)
  getAppointmentHistory: async (params = {}) => {
    return await api.get('/appointments/history/list', { params });
  },
};

// Named exports & convenient wrappers
export const getAllAppointments = appointmentService.getAppointments;

// getAppointmentsByPatient: Get current logged-in patient's appointments
export const getAppointmentsByPatient = async (params = {}) => {
  try {
    return await api.get('/appointments/me', { params });
  } catch (error) {
    // Fallback to /appointments with patientId if /appointments/me doesn't work
    // eslint-disable-next-line no-console
    console.error('Failed to get /appointments/me, trying alternative endpoint:', error);
    const p = { ...(params || {}), current: true };
    return await appointmentService.getAppointments(p);
  }
};

export const getAppointmentById = appointmentService.getAppointmentById;
export const createAppointment = appointmentService.createAppointment;
export const updateAppointment = appointmentService.updateAppointment;
export const confirmAppointment = appointmentService.confirmAppointment;
export const checkInAppointment = appointmentService.checkInAppointment;
export const startAppointment = appointmentService.startAppointment;
export const completeAppointment = appointmentService.completeAppointment;
export const markNoShow = appointmentService.markNoShow;
export const cancelAppointment = appointmentService.cancelAppointment;
export const deleteAppointment = appointmentService.deleteAppointment;
export const getAppointmentHistory = appointmentService.getAppointmentHistory;
export const getAllAppointmentsByPatient = appointmentService.getAllAppointmentsByPatient;

export default appointmentService;
