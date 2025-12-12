const labOrderService = require('./lab_orders.service');
const { successResponse } = require('../../utils/response');

class LabOrderController {
  async createLabOrder(req, res, next) {
    try {
      const labOrder = await labOrderService.createLabOrder(req.validatedBody, req.user);
      return successResponse(res, labOrder, 'Lab order created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async getLabOrders(req, res, next) {
    try {
      const { patient_id, doctor_id, status, page = 1, limit = 10 } = req.query;
      const result = await labOrderService.getLabOrders({
        patient_id,
        doctor_id,
        status,
        page: parseInt(page),
        limit: parseInt(limit),
      }, req.user);
      return successResponse(res, result, 'Lab orders retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getLabOrderById(req, res, next) {
    try {
      const labOrder = await labOrderService.getLabOrderById(req.params.id, req.user);
      return successResponse(res, labOrder, 'Lab order retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateLabOrder(req, res, next) {
    try {
      const labOrder = await labOrderService.updateLabOrder(
        req.params.id,
        req.validatedBody,
        req.user
      );
      return successResponse(res, labOrder, 'Lab order updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateLabResults(req, res, next) {
    try {
      const labOrder = await labOrderService.updateLabResults(
        req.params.id,
        req.validatedBody,
        req.user
      );
      return successResponse(res, labOrder, 'Lab results updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async completeLabOrder(req, res, next) {
    try {
      const labOrder = await labOrderService.completeLabOrder(req.params.id, req.user);
      return successResponse(res, labOrder, 'Lab order completed successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteLabOrder(req, res, next) {
    try {
      await labOrderService.deleteLabOrder(req.params.id, req.user);
      return successResponse(res, null, 'Lab order deleted successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new LabOrderController();