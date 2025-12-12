const express = require('express');
const router = express.Router();
const usersController = require('./users.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');
const { 
  createUserSchema, 
  updateUserSchema,
  updateProfileSchema,
  changePasswordSchema,
  resetPasswordSchema,
  toggleUserStatusSchema,
  assignRoleSchema
} = require('./users.validator');
const { uploadSingle } = require('../../middlewares/upload.middleware');

router.use(authenticate);

// Get all users (Admin only)
router.get('/', 
  authorize(['Admin']), 
  usersController.getAllUsers
);

// Get user stats (Admin only)
router.get('/stats', 
  authorize(['Admin']), 
  usersController.getUserStats
);

// Get users by role (Admin only)
router.get('/role/:role', 
  authorize(['Admin']), 
  usersController.getUsersByRole
);

// Get my profile (All authenticated users)
router.get('/me/profile', 
  usersController.getMyProfile
);

// Update my profile (All authenticated users)
router.put('/me/profile', 
  validate(updateProfileSchema), 
  usersController.updateMyProfile
);

// Change my password (All authenticated users)
router.put('/me/change-password', 
  validate(changePasswordSchema), 
  usersController.changePassword
);

// Get user by ID (Admin only)
router.get('/:id', 
  authorize(['Admin']), 
  usersController.getUserById
);

// Create user (Admin only)
router.post('/', 
  authorize(['Admin']), 
  validate(createUserSchema), 
  usersController.createUser
);

// Update user (Admin only)
router.put('/:id', 
  authorize(['Admin']), 
  validate(updateUserSchema), 
  usersController.updateUser
);

// Toggle user status (Admin only)
router.patch('/:id/status', 
  authorize(['Admin']), 
  validate(toggleUserStatusSchema), 
  usersController.toggleUserStatus
);

// Assign role (Admin only)
router.patch('/:id/role', 
  authorize(['Admin']), 
  validate(assignRoleSchema), 
  usersController.assignRole
);

// Reset user password (Admin only)
router.post('/:id/reset-password', 
  authorize(['Admin']), 
  validate(resetPasswordSchema), 
  usersController.resetPassword
);

// Upload avatar (Admin only)
router.put('/:id/avatar',
  authorize(['Admin']),
  uploadSingle('avatar'),
  usersController.uploadAvatar
);

// Delete user (Admin only)
router.delete('/:id', 
  authorize(['Admin']), 
  usersController.deleteUser
);

// Restore soft-deleted user (Admin only)
router.post('/:id/restore',
  authorize(['Admin']),
  usersController.restoreUser
);

module.exports = router;