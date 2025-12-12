// src/services/userService.js
import api from './api';

export const userService = {
  // Get all users (Admin)
  getAllUsers: async (params = {}) => {
    return await api.get('/users', { params });
  },

  // Get user stats (Admin)
  getUserStats: async () => {
    return await api.get('/users/stats');
  },

  // Get users by role (Admin)
  getUsersByRole: async (role, params = {}) => {
    return await api.get(`/users/role/${role}`, { params });
  },

  // Get my profile
  getMyProfile: async () => {
    return await api.get('/users/me/profile');
  },

  // Update my profile
  updateMyProfile: async (data) => {
    return await api.put('/users/me/profile', data);
  },

  // Change my password
  changeMyPassword: async (currentPassword, newPassword) => {
    return await api.put('/users/me/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  },

  // Get user by ID (Admin)
  getUserById: async (id) => {
    return await api.get(`/users/${id}`);
  },

  // Create user (Admin)
  createUser: async (data) => {
    return await api.post('/users', data);
  },

  // Update user (Admin)
  updateUser: async (id, data) => {
    return await api.put(`/users/${id}`, data);
  },

  // Toggle user status (Admin)
  toggleUserStatus: async (id, isActive) => {
    return await api.patch(`/users/${id}/status`, { is_active: isActive });
  },

  // Assign role (Admin)
  assignRole: async (id, roleId) => {
    return await api.patch(`/users/${id}/role`, { role_id: roleId });
  },

  // Reset user password (Admin)
  resetUserPassword: async (id, newPassword) => {
    return await api.post(`/users/${id}/reset-password`, {
      new_password: newPassword,
    });
  },

  // Delete user (Admin)
  deleteUser: async (id) => {
    return await api.delete(`/users/${id}`);
  },
};

// --- Named exports (so existing imports like `import { getAllUsers } from '.../userService'` work) ---
export const getAllUsers = userService.getAllUsers;
export const getUserStats = userService.getUserStats;
export const getUsersByRole = userService.getUsersByRole;
export const getMyProfile = userService.getMyProfile;
export const updateMyProfile = userService.updateMyProfile;
export const changeMyPassword = userService.changeMyPassword;
export const getUserById = userService.getUserById;
export const createUser = userService.createUser;
export const updateUser = userService.updateUser;
export const toggleUserStatus = userService.toggleUserStatus;
export const assignRole = userService.assignRole;
export const resetUserPassword = userService.resetUserPassword;
export const deleteUser = userService.deleteUser;

// Keep default-ish export for consumers that import the whole service object
export default userService;
