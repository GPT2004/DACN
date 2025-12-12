/* eslint-disable no-console */
const prisma = require('../../config/database');

class AuditLogsService {
  async getAllAuditLogs(filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {};

      if (filters.user_id) {
        where.user_id = parseInt(filters.user_id);
      }

      if (filters.action) {
        where.action = {
          contains: filters.action,
          mode: 'insensitive'
        };
      }

      if (filters.startDate || filters.endDate) {
        where.created_at = {};
        if (filters.startDate) {
          where.created_at.gte = new Date(filters.startDate);
        }
        if (filters.endDate) {
          where.created_at.lte = new Date(filters.endDate);
        }
      }

      const [total, logs] = await Promise.all([
        prisma.audit_logs.count({ where }),
        prisma.audit_logs.findMany({
          where,
          skip,
          take: limit,
          select: {
            id: true,
            user_id: true,
            action: true,
            meta: true,
            created_at: true
          },
          orderBy: {
            created_at: 'desc'
          }
        })
      ]);

      // JOIN THỦ CÔNG: LẤY THÔNG TIN USER
      const userIds = [...new Set(logs.map(log => log.user_id).filter(Boolean))];
      let userMap = {};
      if (userIds.length > 0) {
        const users = await prisma.users.findMany({
          where: { id: { in: userIds } },
          select: {
            id: true,
            full_name: true,
            email: true,
            role: {
              select: { name: true }
            }
          }
        });
        userMap = Object.fromEntries(users.map(u => [u.id, u]));
      }

      const enrichedLogs = logs.map(log => ({
        ...log,
        user: log.user_id ? userMap[log.user_id] || null : null
      }));

      return {
        logs: enrichedLogs,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getAllAuditLogs:', error);
      throw error;
    }
  }

  async getAuditLogById(id) {
    try {
      const log = await prisma.audit_logs.findUnique({
        where: { id: parseInt(id) },
        select: {
          id: true,
          user_id: true,
          action: true,
          meta: true,
          created_at: true
        }
      });

      if (!log) return null;

      // JOIN THỦ CÔNG CHO 1 LOG
      let user = null;
      if (log.user_id) {
        user = await prisma.users.findUnique({
          where: { id: log.user_id },
          select: {
            id: true,
            full_name: true,
            email: true,
            role: {
              select: { name: true }
            }
          }
        });
      }

      return {
        ...log,
        user
      };
    } catch (error) {
      console.error('Error in getAuditLogById:', error);
      throw error;
    }
  }

  async createAuditLog(logData) {
    try {
      const { user_id, action, meta } = logData;

      const log = await prisma.audit_logs.create({
        data: {
          user_id,
          action,
          meta: meta || {}
        },
        select: {
          id: true,
          user_id: true,
          action: true,
          meta: true,
          created_at: true
        }
      });

      // JOIN USER CHO LOG MỚI TẠO
      let user = null;
      if (log.user_id) {
        user = await prisma.users.findUnique({
          where: { id: log.user_id },
          select: {
            full_name: true,
            email: true
          }
        });
      }

      return {
        ...log,
        user
      };
    } catch (error) {
      console.error('Error in createAuditLog:', error);
      throw error;
    }
  }

  async getUserAuditLogs(userId, filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = { user_id: parseInt(userId) };

      if (filters.action) {
        where.action = {
          contains: filters.action,
          mode: 'insensitive'
        };
      }

      if (filters.startDate || filters.endDate) {
        where.created_at = {};
        if (filters.startDate) {
          where.created_at.gte = new Date(filters.startDate);
        }
        if (filters.endDate) {
          where.created_at.lte = new Date(filters.endDate);
        }
      }

      const [total, logs] = await Promise.all([
        prisma.audit_logs.count({ where }),
        prisma.audit_logs.findMany({
          where,
          skip,
          take: limit,
          select: {
            id: true,
            user_id: true,
            action: true,
            meta: true,
            created_at: true
          },
          orderBy: {
            created_at: 'desc'
          }
        })
      ]);

      // JOIN USER (chỉ 1 user)
      let user = null;
      if (logs.length > 0 && logs[0].user_id) {
        user = await prisma.users.findUnique({
          where: { id: logs[0].user_id },
          select: {
            id: true,
            full_name: true,
            email: true,
            role: { select: { name: true } }
          }
        });
      }

      const enrichedLogs = logs.map(log => ({
        ...log,
        user: user ? { id: user.id, full_name: user.full_name, email: user.email, role: user.role } : null
      }));

      return {
        logs: enrichedLogs,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getUserAuditLogs:', error);
      throw error;
    }
  }

  async getActionTypes() {
    try {
      const actions = await prisma.audit_logs.findMany({
        select: { action: true },
        distinct: ['action'],
        orderBy: { action: 'asc' }
      });

      return actions.map(a => a.action);
    } catch (error) {
      console.error('Error in getActionTypes:', error);
      throw error;
    }
  }

  async deleteOldLogs(days = 90) {
    try {
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - days);

      const result = await prisma.audit_logs.deleteMany({
        where: {
          created_at: { lt: cutoffDate }
        }
      });

      return {
        deleted: result.count,
        cutoffDate
      };
    } catch (error) {
      console.error('Error in deleteOldLogs:', error);
      throw error;
    }
  }

  // Helper method to log actions (used by other services)
  async logAction(userId, action, meta = {}) {
    try {
      await this.createAuditLog({
        user_id: userId,
        action,
        meta
      });
    } catch (error) {
      console.error('Error logging action:', error);
      // Don't throw - audit logging should not break main functionality
    }
  }
}

module.exports = new AuditLogsService();