const prisma = require('../config/database');
const emailService = require('../utils/email');
const logger = require('../utils/logger');

class AppointmentRemindersJob {
  async run() {
    try {
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      tomorrow.setHours(0, 0, 0, 0);

      const nextDay = new Date(tomorrow);
      nextDay.setDate(nextDay.getDate() + 1);

      // Get appointments for tomorrow
      const appointments = await prisma.appointments.findMany({
        where: {
          appointment_date: {
            gte: tomorrow,
            lt: nextDay,
          },
          status: {
            in: ['CONFIRMED', 'PENDING'],
          },
        },
        include: {
          patient: {
            include: { user: true },
          },
          doctor: {
            include: { user: true },
          },
        },
      });

      logger.info(`Found ${appointments.length} appointments for tomorrow`);

      // Send notifications
      const notifications = [];
      for (const appointment of appointments) {
        // Create notification in database
        notifications.push({
          user_id: appointment.patient.user_id,
          type: 'APPOINTMENT_REMINDER',
          payload: {
            message: 'Nhắc nhở: Bạn có lịch hẹn vào ngày mai',
            appointment_id: appointment.id,
            doctor_name: appointment.doctor.user.full_name,
            date: appointment.appointment_date,
            time: appointment.appointment_time,
          },
        });

        // Send email reminder
        try {
          await emailService.sendAppointmentReminder(
            appointment,
            appointment.patient.user.email
          );
        } catch (emailError) {
          logger.error(`Failed to send email reminder for appointment ${appointment.id}:`, emailError);
        }
      }

      if (notifications.length > 0) {
        await prisma.notifications.createMany({ data: notifications });
      }

      logger.info(`Sent ${notifications.length} appointment reminders`);
    } catch (error) {
      logger.error('Appointment reminders job error:', error);
    }
  }
}

module.exports = new AppointmentRemindersJob();
