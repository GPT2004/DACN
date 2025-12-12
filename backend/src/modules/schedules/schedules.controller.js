const scheduleService = require('./schedules.service');
const { successResponse } = require('../../utils/response');

class ScheduleController {
  async createSchedule(req, res, next) {
    try {
      const schedule = await scheduleService.createSchedule(req.validatedBody, req.user);
      return successResponse(res, schedule, 'Schedule created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async getSchedules(req, res, next) {
    try {
      const { doctor_id, date, page = 1, limit = 10 } = req.query;
      const result = await scheduleService.getSchedules({
        doctor_id,
        date,
        page: parseInt(page),
        limit: parseInt(limit),
      }, req.user);
      return successResponse(res, result, 'Schedules retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getScheduleById(req, res, next) {
    try {
      const schedule = await scheduleService.getScheduleById(req.params.id);
      return successResponse(res, schedule, 'Schedule retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateSchedule(req, res, next) {
    try {
      const schedule = await scheduleService.updateSchedule(req.params.id, req.validatedBody);
      return successResponse(res, schedule, 'Schedule updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteSchedule(req, res, next) {
    try {
      await scheduleService.deleteSchedule(req.params.id);
      return successResponse(res, null, 'Schedule deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  // Timeslots
  async getAvailableTimeslots(req, res, next) {
    try {
      const { doctor_id, date } = req.query;
      const timeslots = await scheduleService.getAvailableTimeslots(doctor_id, date);
      return successResponse(res, timeslots, 'Available timeslots retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async createTimeslot(req, res, next) {
    try {
      const timeslot = await scheduleService.createTimeslot(req.validatedBody);
      return successResponse(res, timeslot, 'Timeslot created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async updateTimeslot(req, res, next) {
    try {
      const timeslot = await scheduleService.updateTimeslot(req.params.id, req.validatedBody);
      return successResponse(res, timeslot, 'Timeslot updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteTimeslot(req, res, next) {
    try {
      await scheduleService.deleteTimeslot(req.params.id);
      return successResponse(res, null, 'Timeslot deleted successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new ScheduleController();