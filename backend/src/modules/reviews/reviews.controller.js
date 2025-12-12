const ReviewsService = require('./reviews.service');

// Wrapper để tránh lỗi async trong Express
const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res)).catch(next);
};

module.exports = {
  createReview: asyncHandler(async (req, res) => {
    const { doctorId, rating, comment } = req.body;
    const patientId = req.user?.patients?.id;

    if (!doctorId || !rating || !patientId) {
      return res.status(400).json({
        success: false,
        message: 'Missing required fields: doctorId, rating, patientId',
      });
    }

    if (rating < 1 || rating > 5) {
      return res.status(400).json({
        success: false,
        message: 'Rating must be between 1 and 5',
      });
    }

    const review = await ReviewsService.createReview(doctorId, patientId, rating, comment);

    return res.status(200).json({
      success: true,
      message: 'Review created/updated successfully',
      data: review,
    });
  }),

  getDoctorReviews: asyncHandler(async (req, res) => {
    const { doctorId } = req.params;
    const reviews = await ReviewsService.getDoctorReviews(parseInt(doctorId));

    return res.status(200).json({
      success: true,
      data: reviews,
    });
  }),

  getDoctorReviewStats: asyncHandler(async (req, res) => {
    const { doctorId } = req.params;
    const stats = await ReviewsService.getDoctorReviewStats(parseInt(doctorId));

    return res.status(200).json({
      success: true,
      data: stats,
    });
  }),

  getUserReviewForDoctor: asyncHandler(async (req, res) => {
    const { doctorId } = req.params;
    const patientId = req.user?.patients?.id;

    if (!patientId) {
      return res.status(401).json({
        success: false,
        message: 'User is not authenticated as patient',
      });
    }

    const review = await ReviewsService.getUserReviewForDoctor(parseInt(doctorId), patientId);

    return res.status(200).json({
      success: true,
      data: review,
    });
  }),

  deleteReview: asyncHandler(async (req, res) => {
    const { doctorId } = req.params;
    const patientId = req.user?.patients?.id;

    if (!patientId) {
      return res.status(401).json({
        success: false,
        message: 'User is not authenticated as patient',
      });
    }

    await ReviewsService.deleteReview(parseInt(doctorId), patientId);

    return res.status(200).json({
      success: true,
      message: 'Review deleted successfully',
    });
  }),
};
