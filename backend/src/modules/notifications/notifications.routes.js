const express = require('express');
const router = express.Router();
const notificationsController = require('./notifications.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const { createNotificationSchema } = require('./notifications.validator');

router.use(authenticate);

// Get my notifications
router.get('/', notificationsController.getMyNotifications);

// Get unread count
router.get('/unread/count', notificationsController.getUnreadCount);

// Get notification by ID
router.get('/:id', notificationsController.getNotificationById);

// Mark as read
router.patch('/:id/read', notificationsController.markAsRead);

// Mark all as read
router.patch('/read-all', notificationsController.markAllAsRead);

// Delete notification
router.delete('/:id', notificationsController.deleteNotification);

// Admin: Send notification
router.post('/send', authorize(['Admin', 'Receptionist']), validate(createNotificationSchema), notificationsController.sendNotification);

// Admin: Broadcast notification
router.post('/broadcast', authorize(['Admin']), validate(createNotificationSchema), notificationsController.broadcastNotification);

module.exports = router;