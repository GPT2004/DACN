/* eslint-disable no-console */
const prisma = require('../../config/database');

class NotificationsService {
  async getUserNotifications(userId, filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = { user_id: userId };

      if (filters.is_read !== undefined) {
        where.is_read = filters.is_read;
      }

      if (filters.type) {
        where.type = filters.type;
      }

      const [notifications, total] = await Promise.all([
        prisma.notifications.findMany({
          where,
          skip,
          take: limit,
          orderBy: { created_at: 'desc' }
        }),
        prisma.notifications.count({ where })
      ]);

      return {
        notifications,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getUserNotifications:', error);
      throw error;
    }
  }

  async getUnreadCount(userId) {
    try {
      const count = await prisma.notifications.count({
        where: {
          user_id: userId,
          is_read: false
        }
      });

      return count;
    } catch (error) {
      console.error('Error in getUnreadCount:', error);
      throw error;
    }
  }

  async getNotificationById(id, userId) {
    try {
      const notification = await prisma.notifications.findFirst({
        where: {
          id,
          user_id: userId
        }
      });

      return notification;
    } catch (error) {
      console.error('Error in getNotificationById:', error);
      throw error;
    }
  }

  async markAsRead(id, userId) {
    try {
      const notification = await prisma.notifications.findFirst({
        where: {
          id,
          user_id: userId
        }
      });

      if (!notification) {
        return null;
      }

      const updated = await prisma.notifications.update({
        where: { id },
        data: { is_read: true }
      });

      return updated;
    } catch (error) {
      console.error('Error in markAsRead:', error);
      throw error;
    }
  }

  async markAllAsRead(userId) {
    try {
      const result = await prisma.notifications.updateMany({
        where: {
          user_id: userId,
          is_read: false
        },
        data: { is_read: true }
      });

      return {
        count: result.count,
        message: `${result.count} notifications marked as read`
      };
    } catch (error) {
      console.error('Error in markAllAsRead:', error);
      throw error;
    }
  }

  async deleteNotification(id, userId) {
    try {
      const notification = await prisma.notifications.findFirst({
        where: {
          id,
          user_id: userId
        }
      });

      if (!notification) {
        throw new Error('Notification not found');
      }

      await prisma.notifications.delete({
        where: { id }
      });
    } catch (error) {
      console.error('Error in deleteNotification:', error);
      throw error;
    }
  }

  async createNotification(data) {
    try {
      const { user_id, type, payload } = data;

      const notification = await prisma.notifications.create({
        data: {
          user_id,
          type,
          payload: payload || {},
          is_read: false
        }
      });

      return notification;
    } catch (error) {
      console.error('Error in createNotification:', error);
      throw error;
    }
  }

  async broadcastNotification(data) {
    try {
      const { role_name, type, payload } = data;

      const role = await prisma.roles.findUnique({
        where: { name: role_name },
        include: {
          users: {
            where: { is_active: true },
            select: { id: true }
          }
        }
      });

      if (!role) {
        throw new Error('Role not found');
      }

      const notificationsData = role.users.map(user => ({
        user_id: user.id,
        type,
        payload: payload || {},
        is_read: false
      }));

      await prisma.notifications.createMany({
        data: notificationsData
      });

      return {
        count: notificationsData.length,
        role: role_name
      };
    } catch (error) {
      console.error('Error in broadcastNotification:', error);
      throw error;
    }
  }

  // Helper: Create appointment reminder
  async createAppointmentReminder(appointmentId) {
    try {
      const appointment = await prisma.appointments.findUnique({
        where: { id: appointmentId },
        include: {
          patients: {
            include: {
              users: true
            }
          },
          doctors: {
            include: {
              users: true
            }
          }
        }
      });

      if (!appointment) {
        throw new Error('Appointment not found');
      }

      // Notification for patient
      await this.createNotification({
        user_id: appointment.patients.user_id,
        type: 'appointment_reminder',
        payload: {
          appointment_id: appointmentId,
          appointment_date: appointment.appointment_date,
          appointment_time: appointment.appointment_time,
          doctor_name: appointment.doctors.users.full_name,
          message: `Nhắc nhở: Bạn có lịch hẹn vào ${appointment.appointment_date} lúc ${appointment.appointment_time}`
        }
      });

      // Notification for doctor
      await this.createNotification({
        user_id: appointment.doctors.user_id,
        type: 'appointment_reminder',
        payload: {
          appointment_id: appointmentId,
          appointment_date: appointment.appointment_date,
          appointment_time: appointment.appointment_time,
          patient_name: appointment.patients.users.full_name,
          message: `Nhắc nhở: Bạn có lịch khám bệnh nhân ${appointment.patients.users.full_name}`
        }
      });
    } catch (error) {
      console.error('Error in createAppointmentReminder:', error);
      throw error;
    }
  }
}

module.exports = new NotificationsService();