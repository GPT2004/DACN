const roomsService = require('./rooms.service');
const { successResponse } = require('../../utils/response');

class RoomsController {
  async getAllRooms(req, res, next) {
    try {
      const { page = 1, limit = 20, type, status } = req.query;

      const filters = {};
      if (type) filters.type = type;
      if (status) filters.status = status;

      const result = await roomsService.getAllRooms(filters, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Rooms retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getRoomById(req, res, next) {
    try {
      const { id } = req.params;
      const room = await roomsService.getRoomById(parseInt(id));

      if (!room) {
        throw new Error('Room not found');
      }

      return successResponse(res, room, 'Room retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getAvailableRooms(req, res, next) {
    try {
      const { date, startTime, endTime, type } = req.query;

      if (!date || !startTime || !endTime) {
        throw new Error('date, startTime, and endTime are required');
      }

      const rooms = await roomsService.getAvailableRooms({
        date,
        startTime,
        endTime,
        type
      });

      return successResponse(res, rooms, 'Available rooms retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getRoomSchedule(req, res, next) {
    try {
      const { id } = req.params;
      const { startDate, endDate } = req.query;

      if (!startDate || !endDate) {
        throw new Error('startDate and endDate are required');
      }

      const schedule = await roomsService.getRoomSchedule(parseInt(id), {
        startDate,
        endDate
      });

      return successResponse(res, schedule, 'Room schedule retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async createRoom(req, res, next) {
    try {
      const roomData = req.validatedBody;
      const newRoom = await roomsService.createRoom(roomData);

      return successResponse(res, newRoom, 'Room created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async updateRoom(req, res, next) {
    try {
      const { id } = req.params;
      const updateData = req.validatedBody;

      const updatedRoom = await roomsService.updateRoom(parseInt(id), updateData);

      if (!updatedRoom) {
        throw new Error('Room not found');
      }

      return successResponse(res, updatedRoom, 'Room updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteRoom(req, res, next) {
    try {
      const { id } = req.params;
      const userId = req.user.id;
      await roomsService.deleteRoom(parseInt(id), userId);

      return successResponse(res, null, 'Room deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateRoomStatus(req, res, next) {
    try {
      const { id } = req.params;
      const { status } = req.validatedBody;

      const updatedRoom = await roomsService.updateRoomStatus(parseInt(id), status);

      if (!updatedRoom) {
        throw new Error('Room not found');
      }

      return successResponse(res, updatedRoom, 'Room status updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async restoreRoom(req, res, next) {
    try {
      const { id } = req.params;
      const userId = req.user.id;

      await roomsService.restoreRoom(parseInt(id), userId);

      return successResponse(res, null, 'Room restored successfully');
    } catch (error) {
      next(error);
    }
  }

  async getDeletedRooms(req, res, next) {
    try {
      const { page = 1, limit = 20, name } = req.query;

      const filters = {};
      if (name) filters.name = name;

      const result = await roomsService.getDeletedRooms(filters, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Deleted rooms retrieved successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new RoomsController();