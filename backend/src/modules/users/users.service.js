/* eslint-disable no-console */
const prisma = require('../../config/database');
const auditLogsService = require('../audit-logs/audit-logs.service');
const { hashPassword, comparePassword } = require('../../utils/bcrypt');

class UsersService {
  async getAllUsers(filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {};

      // Exclude soft-deleted users by default
      if (!filters.include_deleted) {
        where.deleted_at = null;
      }

      if (filters.role) {
        where.role = {
          name: {
            equals: filters.role,
            mode: 'insensitive'
          }
        };
      }

      if (filters.is_active !== undefined) {
        where.is_active = filters.is_active;
      }

      if (filters.search) {
        where.OR = [
          {
            full_name: {
              contains: filters.search,
              mode: 'insensitive'
            }
          },
          {
            email: {
              contains: filters.search,
              mode: 'insensitive'
            }
          },
          {
            phone: {
              contains: filters.search,
              mode: 'insensitive'
            }
          }
        ];
      }

      const [total, users] = await Promise.all([
        prisma.users.count({ where }),
        prisma.users.findMany({
          where,
          skip,
          take: limit,
          select: {
            id: true,
            email: true,
            full_name: true,
            phone: true,
            dob: true,
            avatar_url: true,
            is_active: true,
            role_id: true,
            created_at: true,
            updated_at: true,
            role: {  // ← Fixed: roles → role
              select: {
                id: true,
                name: true,
                description: true
              }
            }
          },
          orderBy: {
            created_at: 'desc'
          }
        })
      ]);

      return {
        users,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getAllUsers:', error);
      throw error;
    }
  }

  async getUserById(id) {
    try {
      const user = await prisma.users.findUnique({
        where: { id },
        select: {
          id: true,
          email: true,
          full_name: true,
          phone: true,
          dob: true,
          avatar_url: true,
          is_active: true,
          role_id: true,
          created_at: true,
          updated_at: true,
          role: {  // ← Fixed: roles → role
            select: {
              id: true,
              name: true,
              description: true
            }
          },
          doctors: true,
          patients: true
        }
      });

      return user;
    } catch (error) {
      console.error('Error in getUserById:', error);
      throw error;
    }
  }

  async createUser(userData) {
    try {
      const {
        email,
        password,
        full_name,
        phone,
        dob,
        role_id,
        avatar_url
      } = userData;

      // Check if email already exists
      const existingUser = await prisma.users.findUnique({
        where: { email }
      });

      if (existingUser) {
        throw new Error('Email already registered');
      }

      // Check if role exists
      const role = await prisma.roles.findUnique({
        where: { id: role_id }
      });

      if (!role) {
        throw new Error('Role not found');
      }

      const hashedPassword = await hashPassword(password);

      const user = await prisma.users.create({
        data: {
          email,
          password: hashedPassword,
          full_name,
          phone,
          dob: dob ? new Date(dob) : null,
          avatar_url,
          role_id,
          is_active: true
        },
        select: {
          id: true,
          email: true,
          full_name: true,
          phone: true,
          dob: true,
          avatar_url: true,
          is_active: true,
          role_id: true,
          created_at: true,
          role: {  // ← Fixed: roles → role
            select: {
              id: true,
              name: true
            }
          }
        }
      });

      return user;
    } catch (error) {
      console.error('Error in createUser:', error);
      throw error;
    }
  }

  async updateUser(id, updateData) {
    try {
      const user = await prisma.users.findUnique({
        where: { id }
      });

      if (!user) {
        return null;
      }

      // Check if email is being changed and if it already exists
      if (updateData.email && updateData.email !== user.email) {
        const existingUser = await prisma.users.findUnique({
          where: { email: updateData.email }
        });

        if (existingUser) {
          throw new Error('Email already in use');
        }
      }

      const updatedUser = await prisma.users.update({
        where: { id },
        data: {
          ...updateData,
          dob: updateData.dob ? new Date(updateData.dob) : undefined,
          updated_at: new Date()
        },
        select: {
          id: true,
          email: true,
          full_name: true,
          phone: true,
          dob: true,
          avatar_url: true,
          is_active: true,
          role_id: true,
          created_at: true,
          updated_at: true,
          role: {  // ← Fixed: roles → role
            select: {
              id: true,
              name: true
            }
          }
        }
      });

      return updatedUser;
    } catch (error) {
      console.error('Error in updateUser:', error);
      throw error;
    }
  }

  async changePassword(userId, currentPassword, newPassword) {
    try {
      const user = await prisma.users.findUnique({
        where: { id: userId }
      });

      if (!user) {
        throw new Error('User not found');
      }

      // Verify current password
      const isPasswordValid = await comparePassword(currentPassword, user.password);

      if (!isPasswordValid) {
        throw new Error('Current password is incorrect');
      }

      // Hash new password
      const hashedPassword = await hashPassword(newPassword);

      await prisma.users.update({
        where: { id: userId },
        data: {
          password: hashedPassword,
          updated_at: new Date()
        }
      });
    } catch (error) {
      console.error('Error in changePassword:', error);
      throw error;
    }
  }

  async resetPassword(userId, newPassword) {
    try {
      const user = await prisma.users.findUnique({
        where: { id: userId }
      });

      if (!user) {
        throw new Error('User not found');
      }

      const hashedPassword = await hashPassword(newPassword);

      await prisma.users.update({
        where: { id: userId },
        data: {
          password: hashedPassword,
          updated_at: new Date()
        }
      });
    } catch (error) {
      console.error('Error in resetPassword:', error);
      throw error;
    }
  }

  async toggleUserStatus(userId, isActive) {
    try {
      const user = await prisma.users.findUnique({
        where: { id: userId }
      });

      if (!user) {
        return null;
      }

      const updatedUser = await prisma.users.update({
        where: { id: userId },
        data: {
          is_active: isActive,
          updated_at: new Date()
        },
        select: {
          id: true,
          email: true,
          full_name: true,
          is_active: true,
          role: {  // ← Fixed: roles → role
            select: {
              name: true
            }
          }
        }
      });

      return updatedUser;
    } catch (error) {
      console.error('Error in toggleUserStatus:', error);
      throw error;
    }
  }

  async assignRole(userId, roleId) {
    try {
      const [user, role] = await Promise.all([
        prisma.users.findUnique({ where: { id: userId } }),
        prisma.roles.findUnique({ where: { id: roleId } })
      ]);

      if (!user) {
        throw new Error('User not found');
      }

      if (!role) {
        throw new Error('Role not found');
      }

      const updatedUser = await prisma.users.update({
        where: { id: userId },
        data: {
          role_id: roleId,
          updated_at: new Date()
        },
        select: {
          id: true,
          email: true,
          full_name: true,
          role_id: true,
          role: {  // ← Fixed: roles → role
            select: {
              id: true,
              name: true,
              description: true
            }
          }
        }
      });

      return updatedUser;
    } catch (error) {
      console.error('Error in assignRole:', error);
      throw error;
    }
  }

  async deleteUser(userId) {
    try {
      const user = await prisma.users.findUnique({
        where: { id: userId }
      });

      if (!user) {
        throw new Error('User not found');
      }

      // Check if user has related data
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: userId }
      });

      if (doctor) {
        const futureAppointments = await prisma.appointments.count({
          where: {
            doctor_id: doctor.id,
            appointment_date: { gte: new Date() },
            status: { in: ['PENDING', 'CONFIRMED', 'CHECKED_IN'] }
          }
        });

        if (futureAppointments > 0) {
          throw new Error('Cannot delete doctor with future appointments');
        }
      }

      const patient = await prisma.patients.findFirst({
        where: { user_id: userId }
      });

      if (patient) {
        const futureAppointments = await prisma.appointments.count({
          where: {
            patient_id: patient.id,
            appointment_date: { gte: new Date() },
            status: { in: ['PENDING', 'CONFIRMED', 'CHECKED_IN'] }
          }
        });

        if (futureAppointments > 0) {
          throw new Error('Cannot delete patient with future appointments');
        }
      }

      // Soft-delete user
      await prisma.users.update({
        where: { id: userId },
        data: { deleted_at: new Date(), is_active: false, updated_at: new Date() }
      });

      try {
        await auditLogsService.logAction(null, 'user.delete', { user_id: userId });
      } catch (e) {}
    } catch (error) {
      console.error('Error in deleteUser:', error);
      throw error;
    }
  }

  async restoreUser(userId) {
    try {
      const user = await prisma.users.findUnique({ where: { id: userId } });
      if (!user) throw new Error('User not found');

      await prisma.users.update({
        where: { id: userId },
        data: { deleted_at: null, is_active: true, updated_at: new Date() }
      });

      try {
        await auditLogsService.logAction(null, 'user.restore', { user_id: userId });
      } catch (e) {}
    } catch (error) {
      console.error('Error in restoreUser:', error);
      throw error;
    }
  }

  async getUsersByRole(roleName, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      // Normalize role name for case-insensitive search
      const normalizedRoleName = roleName.charAt(0).toUpperCase() + roleName.slice(1).toLowerCase();

      const where = {
        role: {
          name: {
            equals: normalizedRoleName,
            mode: 'insensitive'
          }
        },
        is_active: true
      };

      const [total, users] = await Promise.all([
        prisma.users.count({ where }),
        prisma.users.findMany({
          where,
          skip,
          take: limit,
          select: {
            id: true,
            email: true,
            full_name: true,
            phone: true,
            avatar_url: true,
            created_at: true,
            role: {
              select: {
                name: true
              }
            },
            doctors: normalizedRoleName === 'Doctor',
            patients: normalizedRoleName === 'Patient'
          },
          orderBy: {
            full_name: 'asc'
          }
        })
      ]);

      return {
        users,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getUsersByRole:', error);
      throw error;
    }
  }

  async getUserStats() {
    try {
      const [totalUsers, activeUsers, usersByRole] = await Promise.all([
        prisma.users.count(),
        prisma.users.count({ where: { is_active: true } }),
        prisma.users.groupBy({
          by: ['role_id'],
          _count: { role_id: true }
        })
      ]);

      const roles = await prisma.roles.findMany();

      const byRole = usersByRole.reduce((acc, item) => {
        const role = roles.find(r => r.id === item.role_id);
        if (role) {
          acc[role.name] = item._count.role_id;
        }
        return acc;
      }, {});

      return {
        totalUsers,
        activeUsers,
        inactiveUsers: totalUsers - activeUsers,
        byRole
      };
    } catch (error) {
      console.error('Error in getUserStats:', error);
      throw error;
    }
  }
}

module.exports = new UsersService();