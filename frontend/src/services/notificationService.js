// src/services/notificationService.js
import api from './api';

export const notificationService = {
  // Get my notifications
  getMyNotifications: async (params = {}) => {
    return await api.get('/notifications', { params });
  },

  // Get unread count
  getUnreadCount: async () => {
    return await api.get('/notifications/unread/count');
  },

  // Get notification by ID
  getNotificationById: async (id) => {
    return await api.get(`/notifications/${id}`);
  },

  // Mark as read
  markAsRead: async (id) => {
    return await api.patch(`/notifications/${id}/read`);
  },

  // Mark all as read
  markAllAsRead: async () => {
    return await api.patch('/notifications/read-all');
  },

  // Delete notification
  deleteNotification: async (id) => {
    return await api.delete(`/notifications/${id}`);
  },

  // Send notification (Admin/Receptionist)
  sendNotification: async (userId, type, payload) => {
    return await api.post('/notifications/send', {
      user_id: userId,
      type,
      payload,
    });
  },

  // Broadcast notification (Admin)
  broadcastNotification: async (roleName, type, payload) => {
    return await api.post('/notifications/broadcast', {
      role_name: roleName,
      type,
      payload,
    });
  },
};