const reportsService = require('./reports.service');
const { successResponse } = require('../../utils/response');

class ReportsController {
  async getDashboardStats(req, res, next) {
    try {
      const stats = await reportsService.getDashboardStats();
      return successResponse(res, stats, 'Dashboard stats retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getRevenueReport(req, res, next) {
    try {
      const { startDate, endDate, groupBy = 'day' } = req.query;

      if (!startDate || !endDate) {
        throw new Error('startDate and endDate are required');
      }

      const report = await reportsService.getRevenueReport({
        startDate,
        endDate,
        groupBy
      });

      return successResponse(res, report, 'Revenue report retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getAppointmentReport(req, res, next) {
    try {
      const { startDate, endDate, status } = req.query;

      if (!startDate || !endDate) {
        throw new Error('startDate and endDate are required');
      }

      const report = await reportsService.getAppointmentReport({
        startDate,
        endDate,
        status
      });

      return successResponse(res, report, 'Appointment report retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getDoctorPerformance(req, res, next) {
    try {
      const { startDate, endDate, doctorId } = req.query;

      if (!startDate || !endDate) {
        throw new Error('startDate and endDate are required');
      }

      const report = await reportsService.getDoctorPerformance({
        startDate,
        endDate,
        doctorId: doctorId ? parseInt(doctorId) : null
      });

      return successResponse(res, report, 'Doctor performance retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getCommonDiseases(req, res, next) {
    try {
      const { startDate, endDate, limit = 10 } = req.query;

      if (!startDate || !endDate) {
        throw new Error('startDate and endDate are required');
      }

      const report = await reportsService.getCommonDiseases({
        startDate,
        endDate,
        limit: parseInt(limit)
      });

      return successResponse(res, report, 'Common diseases report retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getStockReport(req, res, next) {
    try {
      const report = await reportsService.getStockReport();
      return successResponse(res, report, 'Stock report retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getPatientReport(req, res, next) {
    try {
      const { startDate, endDate } = req.query;

      if (!startDate || !endDate) {
        throw new Error('startDate and endDate are required');
      }

      const report = await reportsService.getPatientReport({
        startDate,
        endDate
      });

      return successResponse(res, report, 'Patient report retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async exportReport(req, res, next) {
    try {
      const { type, format = 'csv', ...params } = req.query;

      if (!type) {
        throw new Error('Report type is required');
      }

      const result = await reportsService.exportReport(type, format, params);

      res.setHeader('Content-Type', format === 'csv' ? 'text/csv' : 'application/json');
      res.setHeader('Content-Disposition', `attachment; filename=report-${type}-${Date.now()}.${format}`);
      
      return res.send(result);
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new ReportsController();
