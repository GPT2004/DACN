/* eslint-disable no-console */
// src/services/api.js
import axios from 'axios';
import {
  getAuthToken,
  removeAuthToken,
  removeRefreshToken,
  removeUserData,
} from '../utils/storage';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // <-- gửi cookie nếu backend dùng cookie-based auth
});

// Request interceptor: attach Bearer token if present
api.interceptors.request.use(
  (config) => {
    const token = getAuthToken();
    if (token) {
      config.headers = config.headers || {};
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: return response.data for convenience,
// and handle 401 centrally (clean storage + redirect to login)
api.interceptors.response.use(
  (response) => {
    // keep same behaviour as before: return response.data
    return response.data;
  },
  (error) => {
    // network errors (no response) and CORS failures show up here as well
    if (!error.response) {
      // network / CORS / timeout
      console.error('API network error or CORS issue:', error.message);
      return Promise.reject(new Error('Network error'));
    }

    const status = error.response.status;
    if (status === 401) {
      // clear auth info & force redirect to login
      removeAuthToken();
      removeRefreshToken();
      removeUserData();
      // give a tiny delay to ensure storage is cleared before navigate
      setTimeout(() => {
        window.location.href = '/login';
      }, 50);
    }

    // forward original error (so caller can inspect error.response)
    return Promise.reject(error);
  }
);

export default api;
