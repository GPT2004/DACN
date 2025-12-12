// src/services/aiService.js
import api from './api';

const AI_SERVICE_URL = process.env.REACT_APP_AI_SERVICE_URL || 'http://localhost:5000';

export const aiService = {
  // Symptom Checker
  checkSymptoms: async (symptoms) => {
    return await api.post(`${AI_SERVICE_URL}/symptom-checker`, { symptoms });
  },

  // Risk Prediction
  predictRisk: async (patientData) => {
    return await api.post(`${AI_SERVICE_URL}/risk-prediction`, patientData);
  },

  // Analyze medical image (nếu có)
  analyzeMedicalImage: async (imageFile) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    return await api.post(`${AI_SERVICE_URL}/analyze-image`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
};

// Named exports
export const checkSymptoms = aiService.checkSymptoms;
export const predictRisk = aiService.predictRisk;
export const analyzeMedicalImage = aiService.analyzeMedicalImage;

export default aiService;
