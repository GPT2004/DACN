const axios = require('axios');

const AI_SYMPTOM_URL = process.env.AI_SYMPTOM_SERVICE_URL || 'http://localhost:5001';
const AI_COMBINED_URL = process.env.AI_COMBINED_SERVICE_URL || 'http://localhost:5000';

/**
 * Predict department from symptoms
 * @param {string} symptoms - Comma-separated symptoms in Vietnamese
 * @returns {Promise<Object>}
 */
async function predictDepartment(symptoms) {
  try {
    const response = await axios.post(`${AI_SYMPTOM_URL}/api/symptom`, {
      symptoms: symptoms
    }, {
      timeout: 30000,
      headers: { 'Content-Type': 'application/json' }
    });
    
    return {
      success: true,
      data: response.data
    };
  } catch (error) {
    console.error('AI Symptom Service Error:', error.message);
    throw new Error(`Failed to predict department: ${error.message}`);
  }
}

/**
 * Predict chronic disease risk (heart disease & diabetes)
 * @param {Object} healthData - Health metrics data
 * @returns {Promise<Object>}
 */
async function predictChronicDisease(healthData) {
  try {
    // Build form data for the combined API
    const formData = new URLSearchParams();
    
    // Heart-related fields
    if (healthData.Age) formData.append('Age', healthData.Age);
    if (healthData.Sex) formData.append('Sex', healthData.Sex);
    if (healthData.ChestPainType) formData.append('ChestPainType', healthData.ChestPainType);
    if (healthData.RestingBP) formData.append('RestingBP', healthData.RestingBP);
    if (healthData.Cholesterol) formData.append('Cholesterol', healthData.Cholesterol);
    if (healthData.FastingBS !== undefined) formData.append('FastingBS', healthData.FastingBS);
    if (healthData.RestingECG) formData.append('RestingECG', healthData.RestingECG);
    if (healthData.MaxHR) formData.append('MaxHR', healthData.MaxHR);
    if (healthData.ExerciseAngina) formData.append('ExerciseAngina', healthData.ExerciseAngina);
    if (healthData.Oldpeak) formData.append('Oldpeak', healthData.Oldpeak);
    if (healthData.ST_Slope) formData.append('ST_Slope', healthData.ST_Slope);
    
    // Diabetes-related fields
    if (healthData.gender) formData.append('gender', healthData.gender);
    if (healthData.age) formData.append('age', healthData.age);
    if (healthData.hypertension !== undefined) formData.append('hypertension', healthData.hypertension);
    if (healthData.heart_disease !== undefined) formData.append('heart_disease', healthData.heart_disease);
    if (healthData.smoking_history) formData.append('smoking_history', healthData.smoking_history);
    if (healthData.bmi) formData.append('bmi', healthData.bmi);
    if (healthData.HbA1c_level) formData.append('HbA1c_level', healthData.HbA1c_level);
    if (healthData.blood_glucose_level) formData.append('blood_glucose_level', healthData.blood_glucose_level);

    const response = await axios.post(`${AI_COMBINED_URL}/`, formData.toString(), {
      timeout: 30000,
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
    
    // Parse HTML response to extract result
    const html = response.data;
    
    // Simple parsing - you may need to adjust based on actual HTML structure
    // For now, return raw HTML or implement proper parsing
    return {
      success: true,
      html: html,
      message: 'Combined prediction completed. Parse HTML response in frontend.'
    };
  } catch (error) {
    console.error('AI Combined Service Error:', error.message);
    throw new Error(`Failed to predict chronic disease: ${error.message}`);
  }
}

module.exports = {
  predictDepartment,
  predictChronicDisease
};
