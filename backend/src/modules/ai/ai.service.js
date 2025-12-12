const logger = require('../../utils/logger');
const { predictSymptoms } = require('../../services/symptom-predictor');
const { predictCombined } = require('../../services/combined-predictor');

/**
 * Check AI service health
 */
const checkAIServiceHealth = async () => {
  try {
    return {
      status: 'healthy',
      service: 'local-python',
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    logger.error(`Health check failed: ${error.message}`);
    const err = new Error('AI service is unavailable');
    err.code = 'SERVICE_UNAVAILABLE';
    throw err;
  }
};

/**
 * Get status of all prediction models
 */
const getModelStatus = async () => {
  try {
    return {
      symptom_model: 'loaded',
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    logger.error(`Cannot get model status: ${error.message}`);
    const err = new Error('Cannot retrieve model status');
    err.code = 'SERVICE_UNAVAILABLE';
    throw err;
  }
};

/**
 * Predict disease from symptoms - gọi trực tiếp Python script
 * @param {Object} data - { symptoms: string[], patientAge: number, patientGender: string }
 */
const predictDisease = async (data) => {
  try {
    const { symptoms, patientAge, patientGender } = data;
    
    // Gọi Python script trực tiếp
    const result = predictSymptoms(symptoms);  // symptoms is already an array
    
    // Nếu Python không success, throw error
    if (!result.success) {
      throw new Error(result.error || 'Prediction failed');
    }
    
    return result;
  } catch (error) {
    logger.error(`Disease prediction error: ${error.message}`);
    throw new Error(error.message || 'Prediction failed');
  }
};

/**
 * Predict chronic disease risk (diabetes and cardiovascular)
 * @param {Object} data - Medical metrics
 */
const predictChronicDisease = async (data) => {
  try {
    // Gọi combined predictor
    const result = predictCombined(data);
    
    if (!result.success) {
      throw new Error(result.error || 'Prediction failed');
    }
    
    return result;
  } catch (error) {
    logger.error(`Chronic disease prediction error: ${error.message}`);
    throw new Error(error.message || 'Prediction failed');
  }
};

module.exports = {
  checkAIServiceHealth,
  getModelStatus,
  predictDisease,
  predictChronicDisease
};
