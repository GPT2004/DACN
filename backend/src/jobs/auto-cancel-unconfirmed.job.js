const prisma = require('../config/database');
const emailService = require('../utils/email');
const logger = require('../utils/logger');

class AutoCancelUnconfirmedJob {
  async run() {
    try {
      const now = new Date();
      const cutoff = new Date(now.getTime() - 15 * 60 * 1000); // 15 minutes ago

      // Find appointments that are still PENDING, not confirmed by patient, created more than 15 minutes ago
      const toCancel = await prisma.appointments.findMany({
        where: {
          status: 'PENDING',
          patient_confirmed: false,
          created_at: { lt: cutoff }
        },
        include: {
          patient: { include: { user: true } },
          doctor: { include: { user: true } }
        }
      });

      if (toCancel.length === 0) {
        logger.info('No unconfirmed appointments to auto-cancel');
        return;
      }

      logger.info(`Auto-cancelling ${toCancel.length} unconfirmed appointments`);

      for (const appt of toCancel) {
        try {
          await prisma.appointments.update({
            where: { id: appt.id },
            data: { status: 'CANCELLED', updated_at: new Date() }
          });

          // send cancellation email
          try {
            const patientEmail = appt.patient?.user?.email || appt.patient?.email;
            if (patientEmail) {
              await emailService.sendAppointmentCancellation(appt, patientEmail, 'không xác nhận trong vòng 15 phút');
            }
          } catch (e) {
            logger.error(`Failed to send cancellation email for appointment ${appt.id}:`, e);
          }

          // create audit log
          try {
            await prisma.audit_logs.create({ data: { user_id: appt.patient?.user_id || null, action: 'appointment:auto_cancel', meta: { appointment_id: appt.id, reason: 'no_confirmation_15min' } } });
          } catch (e) {
            logger.error('Failed to create audit log for auto-cancel:', e);
          }

        } catch (err) {
          logger.error(`Failed to cancel appointment ${appt.id}:`, err);
        }
      }

    } catch (error) {
      logger.error('Auto-cancel unconfirmed job error:', error);
    }
  }
}

module.exports = new AutoCancelUnconfirmedJob();
