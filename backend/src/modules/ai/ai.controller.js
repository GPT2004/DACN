const aiService = require('../../services/ai.service');

/**
 * Predict department from symptoms
 */
async function predictDepartment(req, res) {
  try {
    const { symptoms } = req.body;
    
    if (!symptoms || !symptoms.trim()) {
      return res.status(400).json({
        success: false,
        message: 'Symptoms are required'
      });
    }

    const result = await aiService.predictDepartment(symptoms);
    
    return res.status(200).json({
      success: true,
      data: result.data
    });
  } catch (error) {
    console.error('Predict Department Error:', error);
    return res.status(500).json({
      success: false,
      message: error.message || 'Failed to predict department'
    });
  }
}

/**
 * Predict chronic disease risk
 */
async function predictChronicDisease(req, res) {
  try {
    const healthData = req.body;
    
    if (!healthData || Object.keys(healthData).length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Health data is required'
      });
    }

    const result = await aiService.predictChronicDisease(healthData);
    
    return res.status(200).json({
      success: true,
      data: result
    });
  } catch (error) {
    console.error('Predict Chronic Disease Error:', error);
    return res.status(500).json({
      success: false,
      message: error.message || 'Failed to predict chronic disease'
    });
  }
}

module.exports = {
  predictDepartment,
  predictChronicDisease
};
