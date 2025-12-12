const notificationsService = require('./notifications.service');
const { successResponse } = require('../../utils/response');

class NotificationsController {
  async getMyNotifications(req, res, next) {
    try {
      const userId = req.user.id;
      const { is_read, type, page = 1, limit = 20 } = req.query;

      const filters = {};
      if (is_read !== undefined) filters.is_read = is_read === 'true';
      if (type) filters.type = type;

      const result = await notificationsService.getUserNotifications(
        userId,
        filters,
        { page: parseInt(page), limit: parseInt(limit) }
      );

      return successResponse(res, result, 'Notifications retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getUnreadCount(req, res, next) {
    try {
      const userId = req.user.id;
      const count = await notificationsService.getUnreadCount(userId);

      return successResponse(res, { count }, 'Unread count retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getNotificationById(req, res, next) {
    try {
      const { id } = req.params;
      const userId = req.user.id;

      const notification = await notificationsService.getNotificationById(
        parseInt(id),
        userId
      );

      if (!notification) {
        throw new Error('Notification not found');
      }

      return successResponse(res, notification, 'Notification retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async markAsRead(req, res, next) {
    try {
      const { id } = req.params;
      const userId = req.user.id;

      const notification = await notificationsService.markAsRead(
        parseInt(id),
        userId
      );

      if (!notification) {
        throw new Error('Notification not found');
      }

      return successResponse(res, notification, 'Notification marked as read');
    } catch (error) {
      next(error);
    }
  }

  async markAllAsRead(req, res, next) {
    try {
      const userId = req.user.id;
      const result = await notificationsService.markAllAsRead(userId);

      return successResponse(res, result, 'All notifications marked as read');
    } catch (error) {
      next(error);
    }
  }

  async deleteNotification(req, res, next) {
    try {
      const { id } = req.params;
      const userId = req.user.id;

      await notificationsService.deleteNotification(parseInt(id), userId);

      return successResponse(res, null, 'Notification deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async sendNotification(req, res, next) {
    try {
      const { user_id, type, payload } = req.validatedBody;

      const notification = await notificationsService.createNotification({
        user_id,
        type,
        payload
      });

      return successResponse(res, notification, 'Notification sent successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async broadcastNotification(req, res, next) {
    try {
      const { role_name, type, payload } = req.validatedBody;

      const result = await notificationsService.broadcastNotification({
        role_name,
        type,
        payload
      });

      return successResponse(res, result, `Notification sent to ${result.count} users`, 201);
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new NotificationsController();