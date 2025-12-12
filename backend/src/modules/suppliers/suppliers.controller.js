const suppliersService = require('./suppliers.service');
const { successResponse } = require('../../utils/response');

class SuppliersController {
  async getAllSuppliers(req, res, next) {
    try {
      const { page = 1, limit = 20, search } = req.query;

      const filters = {};
      if (search) filters.search = search;

      const result = await suppliersService.getAllSuppliers(filters, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Suppliers retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getSupplierById(req, res, next) {
    try {
      const { id } = req.params;
      const supplier = await suppliersService.getSupplierById(parseInt(id));

      if (!supplier) {
        throw new Error('Supplier not found');
      }

      return successResponse(res, supplier, 'Supplier retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async createSupplier(req, res, next) {
    try {
      const supplierData = req.validatedBody;
      const newSupplier = await suppliersService.createSupplier(supplierData);

      return successResponse(res, newSupplier, 'Supplier created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async updateSupplier(req, res, next) {
    try {
      const { id } = req.params;
      const updateData = req.validatedBody;

      const updatedSupplier = await suppliersService.updateSupplier(
        parseInt(id),
        updateData
      );

      if (!updatedSupplier) {
        throw new Error('Supplier not found');
      }

      return successResponse(res, updatedSupplier, 'Supplier updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteSupplier(req, res, next) {
    try {
      const { id } = req.params;
      await suppliersService.deleteSupplier(parseInt(id));

      return successResponse(res, null, 'Supplier deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async getSupplierStats(req, res, next) {
    try {
      const { id } = req.params;
      const stats = await suppliersService.getSupplierStats(parseInt(id));

      return successResponse(res, stats, 'Supplier stats retrieved successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new SuppliersController();