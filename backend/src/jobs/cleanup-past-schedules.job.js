const prisma = require('../config/database');
const logger = require('../utils/logger');

class CleanupPastSchedulesJob {
  async run() {
    try {
      logger.info('Running cleanup past schedules job...');

      const today = new Date();
      today.setHours(0,0,0,0);

      // Find schedules with date < today
      const oldSchedules = await prisma.schedules.findMany({
        where: {
          date: { lt: today }
        }
      });

      logger.info(`Found ${oldSchedules.length} past schedules to evaluate`);

      for (const s of oldSchedules) {
        try {
          // Find appointments on that schedule date for the doctor
          const appts = await prisma.appointments.findMany({
            where: {
              doctor_id: s.doctor_id,
              appointment_date: s.date
            }
          });

          // If no appointments, or all appointments are final (COMPLETED/CANCELLED/NO_SHOW), safe to delete
          const finalStates = ['COMPLETED', 'CANCELLED', 'NO_SHOW'];
          const hasActive = appts.some(a => !finalStates.includes(a.status));

          if (hasActive) {
            logger.info(`Skipping schedule ${s.id} for doctor ${s.doctor_id} because it has active appointments`);
            continue;
          }

          // Delete timeslots for that doctor/date
          try {
            const deletedSlots = await prisma.timeslots.deleteMany({ where: { doctor_id: s.doctor_id, date: s.date } });
            logger.info(`Deleted ${deletedSlots.count} timeslots for schedule ${s.id}`);
          } catch (e) {
            logger.warn(`Failed to delete timeslots for schedule ${s.id}: ${e.message}`);
          }

          // Delete the schedule
          try {
            await prisma.schedules.delete({ where: { id: s.id } });
            logger.info(`Deleted schedule ${s.id} (doctor ${s.doctor_id}, date ${s.date.toISOString().split('T')[0]})`);
          } catch (e) {
            logger.warn(`Failed to delete schedule ${s.id}: ${e.message}`);
          }
        } catch (innerErr) {
          logger.error(`Error processing schedule ${s.id}: ${innerErr.message}`);
        }
      }

      logger.info('Cleanup past schedules job completed');
    } catch (error) {
      logger.error('Error running cleanup past schedules job: ' + error.message);
    }
  }
}

module.exports = new CleanupPastSchedulesJob();
