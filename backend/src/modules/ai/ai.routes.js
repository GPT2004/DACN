const express = require('express');
const router = express.Router();
const aiController = require('./ai.controller');

/**
 * @route   POST /api/ai/predict/department
 * @desc    Predict department from symptoms
 * @access  Public
 */
router.post('/predict/department', aiController.predictDepartment);

/**
 * @route   POST /api/ai/predict/chronic-disease
 * @desc    Predict chronic disease risk (heart & diabetes)
 * @access  Public
 */
router.post('/predict/chronic-disease', aiController.predictChronicDisease);

module.exports = router;
