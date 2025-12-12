// src/services/roomService.js
import api from './api';

export const roomService = {
  // Get all rooms
  getAllRooms: async (params = {}) => {
    return await api.get('/rooms', { params });
  },

  // Get room by ID
  getRoomById: async (id) => {
    return await api.get(`/rooms/${id}`);
  },

  // Get available rooms
  getAvailableRooms: async (date, startTime, endTime, type) => {
    return await api.get('/rooms/available', {
      params: { date, startTime, endTime, type },
    });
  },

  // Get room schedule
  getRoomSchedule: async (id, startDate, endDate) => {
    return await api.get(`/rooms/${id}/schedule`, {
      params: { startDate, endDate },
    });
  },

  // Create room (Admin)
  createRoom: async (data) => {
    return await api.post('/rooms', data);
  },

  // Update room (Admin)
  updateRoom: async (id, data) => {
    return await api.put(`/rooms/${id}`, data);
  },

  // Update room status (Admin/Receptionist)
  updateRoomStatus: async (id, status) => {
    return await api.patch(`/rooms/${id}/status`, { status });
  },

  // Delete room (Admin)
  deleteRoom: async (id) => {
    return await api.delete(`/rooms/${id}`);
  },
};