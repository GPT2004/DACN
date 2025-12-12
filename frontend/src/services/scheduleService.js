// src/services/scheduleService.js
import api from './api';

export const scheduleService = {
  // Get schedules
  getSchedules: async (params = {}) => {
    return await api.get('/schedules', { params });
  },

  // Get schedule by ID
  getScheduleById: async (id) => {
    return await api.get(`/schedules/${id}`);
  },

  // Create schedule
  createSchedule: async (data) => {
    return await api.post('/schedules', data);
  },

  // Update schedule
  updateSchedule: async (id, data) => {
    return await api.put(`/schedules/${id}`, data);
  },

  // Delete schedule
  deleteSchedule: async (id) => {
    return await api.delete(`/schedules/${id}`);
  },

  // Get available timeslots
  getAvailableTimeslots: async (doctorId, date) => {
    const params = {
      doctor_id: doctorId ? Number(doctorId) : undefined,
      date: date ? String(date) : undefined,
    };
    return await api.get('/schedules/timeslots/available', { params });
  },

  // Create timeslot
  createTimeslot: async (data) => {
    return await api.post('/schedules/timeslots', data);
  },

  // Update timeslot
  updateTimeslot: async (id, data) => {
    return await api.put(`/schedules/timeslots/${id}`, data);
  },

  // Delete timeslot
  deleteTimeslot: async (id) => {
    return await api.delete(`/schedules/timeslots/${id}`);
  },
};

export const timeslotService = {
  // Get all timeslots
  getAllTimeslots: async (params = {}) => {
    return await api.get('/timeslots', { params });
  },

  // Get timeslot by ID
  getTimeslotById: async (id) => {
    return await api.get(`/timeslots/${id}`);
  },

  // Get available timeslots
  getAvailableTimeslots: async (doctorId, date) => {
    const params = {
      doctor_id: doctorId ? Number(doctorId) : undefined,
      date: date ? String(date) : undefined,
    };
    return await api.get('/timeslots/available', { params });
  },

  // Get doctor timeslots
  getDoctorTimeslots: async (doctorId, startDate, endDate) => {
    return await api.get(`/timeslots/doctor/${doctorId}`, {
      params: { startDate, endDate },
    });
  },

  // Create timeslot
  createTimeslot: async (data) => {
    return await api.post('/timeslots', data);
  },

  // Create multiple timeslots
  createMultipleTimeslots: async (data) => {
    return await api.post('/timeslots/bulk', data);
  },

  // Update timeslot
  updateTimeslot: async (id, data) => {
    return await api.put(`/timeslots/${id}`, data);
  },

  // Toggle timeslot status
  toggleTimeslotStatus: async (id, isActive) => {
    return await api.patch(`/timeslots/${id}/status`, { is_active: isActive });
  },

  // Bulk update timeslots
  bulkUpdateTimeslots: async (data) => {
    return await api.patch('/timeslots/bulk/update', data);
  },

  // Delete timeslot
  deleteTimeslot: async (id) => {
    return await api.delete(`/timeslots/${id}`);
  },

  // Get timeslot appointments
  getTimeslotAppointments: async (id) => {
    return await api.get(`/timeslots/${id}/appointments`);
  },
};