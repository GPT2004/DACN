const prisma = require('../config/database');
const logger = require('../utils/logger');

class NoShowMarkerJob {
  async run() {
    try {
      const twoHoursAgo = new Date();
      twoHoursAgo.setHours(twoHoursAgo.getHours() - 2);

      // Find appointments that should be marked as NO_SHOW
      const result = await prisma.appointments.updateMany({
        where: {
          status: {
            in: ['CONFIRMED', 'PENDING'],
          },
          appointment_date: {
            lt: twoHoursAgo,
          },
        },
        data: {
          status: 'NO_SHOW',
          updated_at: new Date(),
        },
      });

      if (result.count > 0) {
        logger.info(`Marked ${result.count} appointments as NO_SHOW`);
      } else {
        logger.info('No appointments to mark as NO_SHOW');
      }
    } catch (error) {
      logger.error('No-show marker job error:', error);
    }
  }
}

module.exports = new NoShowMarkerJob();