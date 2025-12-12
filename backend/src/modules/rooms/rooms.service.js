/* eslint-disable no-console */
const prisma = require('../../config/database');

class RoomsService {
  async getAllRooms(filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {};

      if (filters.type) {
        where.type = {
          contains: filters.type,
          mode: 'insensitive'
        };
      }

      where.deleted_at = null;

      const [total, rooms] = await Promise.all([
        prisma.rooms.count({ where }),
        prisma.rooms.findMany({
          where,
          skip,
          take: limit,
          include: {
            _count: {
              select: {
                schedules: true
              }
            }
          },
          orderBy: {
            name: 'asc'
          }
        })
      ]);

      return {
        rooms,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getAllRooms:', error);
      throw error;
    }
  }

  async getRoomById(id) {
    try {
      const room = await prisma.rooms.findUnique({
        where: { id, deleted_at: null },
        include: {
          schedules: {
            where: {
              date: {
                gte: new Date()
              }
            },
            take: 10,
            orderBy: {
              date: 'asc'
            },
            include: {
              doctors: {
                include: {
                  users: {
                    select: {
                      full_name: true
                    }
                  }
                }
              }
            }
          },
          _count: {
            select: {
              schedules: true
            }
          }
        }
      });

      return room;
    } catch (error) {
      console.error('Error in getRoomById:', error);
      throw error;
    }
  }

  async getAvailableRooms({ date, startTime, endTime, type }) {
    try {
      const targetDate = new Date(date);
      
      const where = {};
      if (type) {
        where.type = type;
      }

      // Get all rooms
      const allRooms = await prisma.rooms.findMany({
        where: {
          ...where,
          deleted_at: null
        },
        orderBy: {
          name: 'asc'
        }
      });

      // Get rooms that have schedules overlapping with requested time
      const busyRooms = await prisma.schedules.findMany({
        where: {
          date: targetDate,
          OR: [
            {
              AND: [
                { start_time: { lte: startTime } },
                { end_time: { gt: startTime } }
              ]
            },
            {
              AND: [
                { start_time: { lt: endTime } },
                { end_time: { gte: endTime } }
              ]
            },
            {
              AND: [
                { start_time: { gte: startTime } },
                { end_time: { lte: endTime } }
              ]
            }
          ]
        },
        select: {
          room_id: true
        }
      });

      const busyRoomIds = busyRooms.map(s => s.room_id);

      // Filter out busy rooms
      const availableRooms = allRooms.filter(room => !busyRoomIds.includes(room.id));

      return availableRooms;
    } catch (error) {
      console.error('Error in getAvailableRooms:', error);
      throw error;
    }
  }

  async getRoomSchedule(roomId, { startDate, endDate }) {
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);

      const schedules = await prisma.schedules.findMany({
        where: {
          room_id: roomId,
          date: {
            gte: start,
            lte: end
          }
        },
        include: {
          doctors: {
            include: {
              users: {
                select: {
                  full_name: true,
                  phone: true
                }
              }
            }
          }
        },
        orderBy: [
          { date: 'asc' },
          { start_time: 'asc' }
        ]
      });

      return schedules;
    } catch (error) {
      console.error('Error in getRoomSchedule:', error);
      throw error;
    }
  }

  async createRoom(roomData) {
    try {
      const { name, type, description, capacity } = roomData;

      // Check if room name already exists
      const existingRoom = await prisma.rooms.findFirst({
        where: {
          name: {
            equals: name,
            mode: 'insensitive'
          }
        }
      });

      if (existingRoom) {
        throw new Error('Room name already exists');
      }

      const room = await prisma.rooms.create({
        data: {
          name,
          type,
          description,
          capacity: capacity || 1
        }
      });

      return room;
    } catch (error) {
      console.error('Error in createRoom:', error);
      throw error;
    }
  }

  async updateRoom(id, updateData) {
    try {
      const room = await prisma.rooms.findUnique({
        where: { id, deleted_at: null }
      });

      if (!room) {
        return null;
      }

      // Check if new name conflicts with existing room
      if (updateData.name && updateData.name !== room.name) {
        const existingRoom = await prisma.rooms.findFirst({
          where: {
            name: {
              equals: updateData.name,
              mode: 'insensitive'
            },
            id: {
              not: id
            }
          }
        });

        if (existingRoom) {
          throw new Error('Room name already exists');
        }
      }

      const updatedRoom = await prisma.rooms.update({
        where: { id },
        data: updateData
      });

      return updatedRoom;
    } catch (error) {
      console.error('Error in updateRoom:', error);
      throw error;
    }
  }

  async deleteRoom(id, userId) {
    try {
      const room = await prisma.rooms.findUnique({
        where: { id, deleted_at: null },
        include: {
          _count: {
            select: {
              schedules: true
            }
          }
        }
      });

      if (!room) {
        throw new Error('Room not found');
      }

      // Check if room has future schedules
      const futureSchedules = await prisma.schedules.count({
        where: {
          room_id: id,
          date: {
            gte: new Date()
          }
        }
      });

      if (futureSchedules > 0) {
        throw new Error('Cannot delete room with future schedules');
      }

      // Soft delete
      await prisma.rooms.update({
        where: { id },
        data: {
          deleted_at: new Date()
        }
      });

      // Log audit
      await prisma.audit_logs.create({
        data: {
          user_id: userId,
          action: 'DELETE_ROOM',
          meta: {
            room_id: id,
            room_name: room.name
          }
        }
      });
    } catch (error) {
      console.error('Error in deleteRoom:', error);
      throw error;
    }
  }

  async updateRoomStatus(id, status) {
    try {
      const room = await prisma.rooms.findUnique({
        where: { id, deleted_at: null }
      });

      if (!room) {
        return null;
      }

      // Note: The schema doesn't have a status field, 
      // but we can add it to description or use type field
      const updatedRoom = await prisma.rooms.update({
        where: { id },
        data: {
          description: status
        }
      });

      return updatedRoom;
    } catch (error) {
      console.error('Error in updateRoomStatus:', error);
      throw error;
    }
  }

  async restoreRoom(id, userId) {
    try {
      const room = await prisma.rooms.findUnique({
        where: { id }
      });

      if (!room) {
        throw new Error('Room not found');
      }

      if (!room.deleted_at) {
        throw new Error('Room is not deleted');
      }

      await prisma.rooms.update({
        where: { id },
        data: {
          deleted_at: null
        }
      });

      // Log audit
      await prisma.audit_logs.create({
        data: {
          user_id: userId,
          action: 'RESTORE_ROOM',
          meta: {
            room_id: id,
            room_name: room.name
          }
        }
      });
    } catch (error) {
      console.error('Error in restoreRoom:', error);
      throw error;
    }
  }

  async getDeletedRooms(filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {
        deleted_at: {
          not: null
        }
      };

      if (filters.name) {
        where.name = {
          contains: filters.name,
          mode: 'insensitive'
        };
      }

      const [total, rooms] = await Promise.all([
        prisma.rooms.count({ where }),
        prisma.rooms.findMany({
          where,
          skip,
          take: limit,
          orderBy: {
            deleted_at: 'desc'
          }
        })
      ]);

      return {
        rooms,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getDeletedRooms:', error);
      throw error;
    }
  }
}

module.exports = new RoomsService();
