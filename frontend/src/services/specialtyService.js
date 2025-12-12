import api from './api';

const specialtyService = {
  getAll: async (params = {}) => {
    try {
      const response = await api.get('/specialties', { params });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getById: async (id) => {
    try {
      const response = await api.get(`/specialties/${id}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  create: async (data) => {
    try {
      if (typeof FormData !== 'undefined' && data instanceof FormData) {
        const response = await api.post('/specialties', data, { headers: { 'Content-Type': 'multipart/form-data' } });
        return response.data;
      }
      const response = await api.post('/specialties', data);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  update: async (id, data) => {
    try {
      if (typeof FormData !== 'undefined' && data instanceof FormData) {
        const response = await api.put(`/specialties/${id}`, data, { headers: { 'Content-Type': 'multipart/form-data' } });
        return response.data;
      }
      const response = await api.put(`/specialties/${id}`, data);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  delete: async (id) => {
    try {
      const response = await api.delete(`/specialties/${id}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  search: async (query, params = {}) => {
    try {
      const response = await api.get('/specialties', { 
        params: { ...params, search: query } 
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  }
};

export default specialtyService;
