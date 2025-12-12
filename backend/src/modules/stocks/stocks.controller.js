const stocksService = require('./stocks.service');
const { successResponse } = require('../../utils/response');

class StocksController {
  async getAllStocks(req, res, next) {
    try {
      const { page = 1, limit = 20, medicine_id, low_stock, expiring } = req.query;

      const filters = {};
      if (medicine_id) filters.medicine_id = parseInt(medicine_id);
      if (low_stock === 'true') filters.low_stock = true;
      if (expiring === 'true') filters.expiring = true;

      const result = await stocksService.getAllStocks(filters, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Stocks retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getStockById(req, res, next) {
    try {
      const { id } = req.params;
      const stock = await stocksService.getStockById(parseInt(id));

      if (!stock) {
        throw new Error('Stock not found');
      }

      return successResponse(res, stock, 'Stock retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getMedicineStock(req, res, next) {
    try {
      const { medicineId } = req.params;
      const stock = await stocksService.getMedicineStock(parseInt(medicineId));

      return successResponse(res, stock, 'Medicine stock retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getLowStockItems(req, res, next) {
    try {
      const { threshold = 20 } = req.query;
      const items = await stocksService.getLowStockItems(parseInt(threshold));

      return successResponse(res, items, 'Low stock items retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getExpiringStock(req, res, next) {
    try {
      const { days = 30 } = req.query;
      const items = await stocksService.getExpiringStock(parseInt(days));

      return successResponse(res, items, 'Expiring stock retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getExpiredStock(req, res, next) {
    try {
      const items = await stocksService.getExpiredStock();
      return successResponse(res, items, 'Expired stock retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async createStock(req, res, next) {
    try {
      const stockData = req.validatedBody;
      const newStock = await stocksService.createStock(stockData);

      return successResponse(res, newStock, 'Stock created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async updateStock(req, res, next) {
    try {
      const { id } = req.params;
      const updateData = req.validatedBody;

      const updatedStock = await stocksService.updateStock(parseInt(id), updateData);

      if (!updatedStock) {
        throw new Error('Stock not found');
      }

      return successResponse(res, updatedStock, 'Stock updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async adjustStockQuantity(req, res, next) {
    try {
      const { id } = req.params;
      const { quantity, reason } = req.validatedBody;
      const userId = req.user.id;

      const result = await stocksService.adjustStockQuantity(
        parseInt(id),
        quantity,
        reason,
        userId
      );

      return successResponse(res, result, 'Stock quantity adjusted successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteStock(req, res, next) {
    try {
      const { id } = req.params;
      await stocksService.deleteStock(parseInt(id));

      return successResponse(res, null, 'Stock deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async getStockHistory(req, res, next) {
    try {
      const { medicineId } = req.params;
      const { page = 1, limit = 20 } = req.query;

      const result = await stocksService.getStockHistory(
        parseInt(medicineId),
        { page: parseInt(page), limit: parseInt(limit) }
      );

      return successResponse(res, result, 'Stock history retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getStockAlerts(req, res, next) {
    try {
      const alerts = await stocksService.getStockAlerts();
      return successResponse(res, alerts, 'Stock alerts retrieved successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new StocksController();