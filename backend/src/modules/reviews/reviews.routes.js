const express = require('express');
const controller = require('./reviews.controller');
const { authenticate } = require('../../middlewares/auth.middleware');

const router = express.Router();

/**
 * POST /api/reviews - Create or update a review
 */
router.post('/', authenticate, controller.createReview);

/**
 * GET /api/reviews/doctor/:doctorId/stats - Get review statistics for a doctor (MUST be before :doctorId)
 */
router.get('/doctor/:doctorId/stats', controller.getDoctorReviewStats);

/**
 * GET /api/reviews/doctor/:doctorId/my-review - Get user's review for a doctor (MUST be before :doctorId)
 */
router.get('/doctor/:doctorId/my-review', authenticate, controller.getUserReviewForDoctor);

/**
 * GET /api/reviews/doctor/:doctorId - Get all reviews for a doctor
 */
router.get('/doctor/:doctorId', controller.getDoctorReviews);

/**
 * DELETE /api/reviews/doctor/:doctorId - Delete a review
 */
router.delete('/doctor/:doctorId', authenticate, controller.deleteReview);

module.exports = router;
