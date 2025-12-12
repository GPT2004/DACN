const usersService = require('./users.service');
const { successResponse } = require('../../utils/response');
const { uploadBuffer } = require('../../utils/cloudinary');

class UsersController {
  async getAllUsers(req, res, next) {
    try {
      const { page = 1, limit = 20, role, search, is_active } = req.query;

      const filters = {};
      if (role) filters.role = role;
      if (search) filters.search = search;
      if (is_active !== undefined) filters.is_active = is_active === 'true';

      const result = await usersService.getAllUsers(filters, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Users retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getUserById(req, res, next) {
    try {
      const { id } = req.params;
      const user = await usersService.getUserById(parseInt(id));

      if (!user) {
        throw new Error('User not found');
      }

      return successResponse(res, user, 'User retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getMyProfile(req, res, next) {
    try {
      const userId = req.user.id;
      const user = await usersService.getUserById(userId);

      if (!user) {
        throw new Error('User not found');
      }

      return successResponse(res, user, 'Profile retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async createUser(req, res, next) {
    try {
      const userData = req.validatedBody;
      const newUser = await usersService.createUser(userData);

      return successResponse(res, newUser, 'User created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async updateUser(req, res, next) {
    try {
      const { id } = req.params;
      const updateData = req.validatedBody;

      const updatedUser = await usersService.updateUser(parseInt(id), updateData);

      if (!updatedUser) {
        throw new Error('User not found');
      }

      return successResponse(res, updatedUser, 'User updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateMyProfile(req, res, next) {
    try {
      const userId = req.user.id;
      const updateData = req.validatedBody;

      const updatedUser = await usersService.updateUser(userId, updateData);

      if (!updatedUser) {
        throw new Error('User not found');
      }

      return successResponse(res, updatedUser, 'Profile updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async changePassword(req, res, next) {
    try {
      const userId = req.user.id;
      const { current_password, new_password } = req.validatedBody;

      await usersService.changePassword(userId, current_password, new_password);

      return successResponse(res, null, 'Password changed successfully');
    } catch (error) {
      next(error);
    }
  }

  async resetPassword(req, res, next) {
    try {
      const { id } = req.params;
      const { new_password } = req.validatedBody;

      await usersService.resetPassword(parseInt(id), new_password);

      return successResponse(res, null, 'Password reset successfully');
    } catch (error) {
      next(error);
    }
  }

  async toggleUserStatus(req, res, next) {
    try {
      const { id } = req.params;
      const { is_active } = req.validatedBody;

      const updatedUser = await usersService.toggleUserStatus(parseInt(id), is_active);

      if (!updatedUser) {
        throw new Error('User not found');
      }

      return successResponse(res, updatedUser, 'User status updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async assignRole(req, res, next) {
    try {
      const { id } = req.params;
      const { role_id } = req.validatedBody;

      const updatedUser = await usersService.assignRole(parseInt(id), parseInt(role_id));

      if (!updatedUser) {
        throw new Error('User not found');
      }

      return successResponse(res, updatedUser, 'Role assigned successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteUser(req, res, next) {
    try {
      const { id } = req.params;
      await usersService.deleteUser(parseInt(id));

      return successResponse(res, null, 'User deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async restoreUser(req, res, next) {
    try {
      const { id } = req.params;
      await usersService.restoreUser(parseInt(id));

      return successResponse(res, null, 'User restored successfully');
    } catch (error) {
      next(error);
    }
  }

  async uploadAvatar(req, res, next) {
    try {
      const { id } = req.params;

      if (!req.file) {
        const err = new Error('No file uploaded');
        err.status = 400;
        throw err;
      }

      const fs = require('fs').promises;
      let buffer;
      if (req.file.buffer) {
        buffer = req.file.buffer;
      } else if (req.file.path) {
        buffer = await fs.readFile(req.file.path);
      }

      if (!buffer) {
        const err = new Error('No file uploaded');
        err.status = 400;
        throw err;
      }

      // Upload to Cloudinary
      const options = {
        folder: process.env.CLOUDINARY_AVATAR_FOLDER || 'avatars',
        resource_type: 'image',
        public_id: `user_${id}_${Date.now()}`
      };

      const result = await uploadBuffer(buffer, options);

      // remove local file if any
      if (req.file.path) {
        try { await fs.unlink(req.file.path); } catch (e) {}
      }

      // Save avatar URL to user
      const updatedUser = await usersService.updateUser(parseInt(id), { avatar_url: result.secure_url });

      if (!updatedUser) {
        const err = new Error('User not found');
        err.status = 404;
        throw err;
      }

      return successResponse(res, { avatar_url: result.secure_url }, 'Avatar uploaded successfully');
    } catch (error) {
      next(error);
    }
  }

  async getUsersByRole(req, res, next) {
    try {
      const { role } = req.params;
      const { page = 1, limit = 20 } = req.query;

      const result = await usersService.getUsersByRole(role, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Users retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getUserStats(req, res, next) {
    try {
      const stats = await usersService.getUserStats();
      return successResponse(res, stats, 'User stats retrieved successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new UsersController();