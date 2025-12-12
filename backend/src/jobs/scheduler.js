const cron = require('node-cron');
const appointmentReminders = require('./appointment-reminders.job');
const stockAlerts = require('./stock-alerts.job');
const noShowMarker = require('./no-show-marker.job');
const cleanupSchedules = require('./cleanup-past-schedules.job');
const logger = require('../utils/logger');

class Scheduler {
  startAll() {
    logger.info('🕐 Initializing scheduled jobs...');

    // Appointment reminders - Every day at 8:00 AM
    cron.schedule('0 8 * * *', () => {
      logger.info('Running appointment reminders job...');
      appointmentReminders.run();
    });

    // Low stock alerts - Every day at 9:00 AM
    cron.schedule('0 9 * * *', () => {
      logger.info('Running low stock alerts job...');
      stockAlerts.checkLowStock();
    });

    // Expiring medicines alerts - Every day at 9:00 AM
    cron.schedule('0 9 * * *', () => {
      logger.info('Running expiring medicines alerts job...');
      stockAlerts.checkExpiringMedicines();
    });

    // Mark no-show appointments - Every hour
    cron.schedule('0 * * * *', () => {
      logger.info('Running no-show marker job...');
      noShowMarker.run();
    });

    // Auto-cancel unconfirmed appointments - Every minute
    cron.schedule('* * * * *', () => {
      logger.info('Running auto-cancel unconfirmed appointments job...');
      try {
        const autoCancel = require('./auto-cancel-unconfirmed.job');
        autoCancel.run();
      } catch (err) {
        logger.error('Failed to run auto-cancel job:', err);
      }
    });

    // Cleanup past schedules - Every day at 00:05
    cron.schedule('5 0 * * *', () => {
      logger.info('Running cleanup past schedules job...');
      cleanupSchedules.run();
    });

    logger.info('✅ All scheduled jobs initialized');
  }
}

module.exports = new Scheduler();