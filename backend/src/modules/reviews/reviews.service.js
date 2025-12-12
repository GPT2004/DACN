const prisma = require('../../config/database');

class ReviewsService {
  /**
   * Create or update a review
   */
  async createReview(doctorId, patientId, rating, comment) {
    try {
      // Upsert: update if exists, create if not
      const review = await prisma.doctor_reviews.upsert({
        where: {
          doctor_id_patient_id: {
            doctor_id: doctorId,
            patient_id: patientId,
          },
        },
        update: {
          rating: Math.min(5, Math.max(1, rating)), // Clamp between 1-5
          comment: comment || null,
          updated_at: new Date(),
        },
        create: {
          doctor_id: doctorId,
          patient_id: patientId,
          rating: Math.min(5, Math.max(1, rating)), // Clamp between 1-5
          comment: comment || null,
        },
        include: {
          patient: {
            include: { user: true },
          },
        },
      });

      // Recalculate doctor's average rating
      await this.updateDoctorRating(doctorId);

      return review;
    } catch (error) {
      console.error('Error creating review:', error);
      throw error;
    }
  }

  /**
   * Get all reviews for a doctor
   */
  async getDoctorReviews(doctorId) {
    try {
      const reviews = await prisma.doctor_reviews.findMany({
        where: { doctor_id: doctorId },
        include: {
          patient: {
            include: { user: true },
          },
        },
        orderBy: { created_at: 'desc' },
      });

      return reviews;
    } catch (error) {
      console.error('Error fetching reviews:', error);
      throw error;
    }
  }

  /**
   * Get review statistics for a doctor
   */
  async getDoctorReviewStats(doctorId) {
    try {
      const reviews = await prisma.doctor_reviews.findMany({
        where: { doctor_id: doctorId },
      });

      if (reviews.length === 0) {
        return {
          total_reviews: 0,
          average_rating: 0,
          rating_distribution: { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
        };
      }

      const distribution = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
      let totalRating = 0;

      reviews.forEach((review) => {
        distribution[review.rating] += 1;
        totalRating += review.rating;
      });

      const averageRating = totalRating / reviews.length;

      return {
        total_reviews: reviews.length,
        average_rating: parseFloat(averageRating.toFixed(1)),
        rating_distribution: distribution,
      };
    } catch (error) {
      console.error('Error fetching review stats:', error);
      throw error;
    }
  }

  /**
   * Update doctor's average rating based on all reviews
   */
  async updateDoctorRating(doctorId) {
    try {
      const reviews = await prisma.doctor_reviews.findMany({
        where: { doctor_id: doctorId },
      });

      let averageRating = 0;
      if (reviews.length > 0) {
        const totalRating = reviews.reduce((sum, r) => sum + r.rating, 0);
        averageRating = totalRating / reviews.length;
      }

      await prisma.doctors.update({
        where: { id: doctorId },
        data: { rating: averageRating },
      });

      return averageRating;
    } catch (error) {
      console.error('Error updating doctor rating:', error);
      throw error;
    }
  }

  /**
   * Get user's review for a specific doctor
   */
  async getUserReviewForDoctor(doctorId, patientId) {
    try {
      const review = await prisma.doctor_reviews.findUnique({
        where: {
          doctor_id_patient_id: {
            doctor_id: doctorId,
            patient_id: patientId,
          },
        },
        include: {
          patient: {
            include: { user: true },
          },
        },
      });

      return review || null;
    } catch (error) {
      console.error('Error fetching user review:', error);
      throw error;
    }
  }

  /**
   * Delete a review
   */
  async deleteReview(doctorId, patientId) {
    try {
      await prisma.doctor_reviews.delete({
        where: {
          doctor_id_patient_id: {
            doctor_id: doctorId,
            patient_id: patientId,
          },
        },
      });

      // Recalculate doctor's average rating
      await this.updateDoctorRating(doctorId);

      return { success: true };
    } catch (error) {
      console.error('Error deleting review:', error);
      throw error;
    }
  }
}

module.exports = new ReviewsService();
