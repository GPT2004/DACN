const medicineService = require('./medicines.service');
const { successResponse } = require('../../utils/response');

class MedicineController {
  // Medicines
  async createMedicine(req, res, next) {
    try {
      const medicine = await medicineService.createMedicine(req.validatedBody);
      return successResponse(res, medicine, 'Medicine created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async getMedicines(req, res, next) {
    try {
      const { search, page = 1, limit = 10 } = req.query;
      const result = await medicineService.getMedicines({
        search,
        page: parseInt(page),
        limit: parseInt(limit),
      });
      return successResponse(res, result, 'Medicines retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getMedicineById(req, res, next) {
    try {
      const medicine = await medicineService.getMedicineById(req.params.id);
      return successResponse(res, medicine, 'Medicine retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateMedicine(req, res, next) {
    try {
      const medicine = await medicineService.updateMedicine(req.params.id, req.validatedBody);
      return successResponse(res, medicine, 'Medicine updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteMedicine(req, res, next) {
    try {
      await medicineService.deleteMedicine(req.params.id);
      return successResponse(res, null, 'Medicine deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async restoreMedicine(req, res, next) {
    try {
      const result = await medicineService.restoreMedicine(req.params.id);
      return successResponse(res, result, 'Medicine restored successfully');
    } catch (error) {
      next(error);
    }
  }

  // Get total stock for a medicine
  async getStockByMedicineId(req, res, next) {
    try {
      const medicine = await medicineService.getMedicineById(req.params.id);
      return successResponse(res, { total_stock: medicine.total_stock }, 'Stock retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  // Stocks
  async createStock(req, res, next) {
    try {
      const stock = await medicineService.createStock(req.validatedBody);
      return successResponse(res, stock, 'Stock created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async getStocks(req, res, next) {
    try {
      const { medicine_id, low_stock, expiring_soon, page = 1, limit = 10 } = req.query;
      const result = await medicineService.getStocks({
        medicine_id,
        low_stock: low_stock === 'true',
        expiring_soon: expiring_soon === 'true',
        page: parseInt(page),
        limit: parseInt(limit),
      });
      return successResponse(res, result, 'Stocks retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getStockById(req, res, next) {
    try {
      const stock = await medicineService.getStockById(req.params.id);
      return successResponse(res, stock, 'Stock retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateStock(req, res, next) {
    try {
      const stock = await medicineService.updateStock(req.params.id, req.validatedBody);
      return successResponse(res, stock, 'Stock updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteStock(req, res, next) {
    try {
      await medicineService.deleteStock(req.params.id);
      return successResponse(res, null, 'Stock deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async getStockSummary(req, res, next) {
    try {
      const summary = await medicineService.getStockSummary();
      return successResponse(res, summary, 'Stock summary retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getLowStockAlerts(req, res, next) {
    try {
      const alerts = await medicineService.getLowStockAlerts();
      return successResponse(res, alerts, 'Low stock alerts retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getExpiringMedicines(req, res, next) {
    try {
      const { days = 30 } = req.query;
      const expiring = await medicineService.getExpiringMedicines(parseInt(days));
      return successResponse(res, expiring, 'Expiring medicines retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async importMedicines(req, res, next) {
    try {
      const { medicines } = req.validatedBody;
      
      if (!Array.isArray(medicines) || medicines.length === 0) {
        return res.status(400).json({
          success: false,
          message: 'Danh sách thuốc không hợp lệ hoặc trống'
        });
      }
      
      const result = await medicineService.importMedicines(medicines);
      return successResponse(res, result, 'Medicines imported successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new MedicineController();