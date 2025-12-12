// src/services/authService.js
import api from './api';

export const authService = {
  // Register new patient
  register: async (userData) => {
    return await api.post('/auth/register', userData);
  },

  // Flexible Login:
  // - login(email, password)
  // - login({ email, password })
  login: async (emailOrPayload, password) => {
    // If first arg is an object and second arg not provided -> treat as payload
    if (typeof emailOrPayload === 'object' && emailOrPayload !== null && (password === undefined || password === null)) {
      const { email, password: pw } = emailOrPayload;
      return await api.post('/auth/login', { email, password: pw });
    }

    // Otherwise assume (email, password)
    return await api.post('/auth/login', { email: emailOrPayload, password });
  },

  // Logout
  logout: async () => {
    return await api.post('/auth/logout');
  },

  // Get current user profile
  getProfile: async () => {
    return await api.get('/auth/me');
  },

  // Change password
  changePassword: async (currentPassword, newPassword) => {
    return await api.put('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  },

  // Refresh token
  refreshToken: async (refreshToken) => {
    return await api.post('/auth/refresh', { refreshToken });
  },

  // Forgot password (added)
  forgotPassword: async (email) => {
    // backend endpoint assumed to be /auth/forgot-password
    return await api.post('/auth/forgot-password', { email });
  },

  checkEmail: async (email) => {
    return await api.get(`/auth/check-email?email=${encodeURIComponent(email)}`);
  },

  // Reset password (if backend supports)
  resetPassword: async (token, newPassword) => {
    return await api.post('/auth/reset-password', { token, new_password: newPassword });
  },
};

// Named exports (aliases)
export const register = authService.register;
export const registerUser = authService.register; // alias used in some files
export const login = authService.login;
export const logout = authService.logout;
export const getProfile = authService.getProfile;
export const changePassword = authService.changePassword;
export const refreshToken = authService.refreshToken;
export const forgotPassword = authService.forgotPassword;
export const resetPassword = authService.resetPassword;
export const checkEmail = authService.checkEmail;

export default authService;
