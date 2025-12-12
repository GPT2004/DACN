/* eslint-disable no-console */
const prisma = require('../../config/database');

class SuppliersService {
  async getAllSuppliers(filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {};

      if (filters.search) {
        where.name = {
          contains: filters.search,
          mode: 'insensitive'
        };
      }

      const [total, suppliers] = await Promise.all([
        prisma.suppliers.count({ where }),
        prisma.suppliers.findMany({
          where,
          skip,
          take: limit,
          orderBy: {
            name: 'asc'
          }
        })
      ]);

      return {
        suppliers,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getAllSuppliers:', error);
      throw error;
    }
  }

  async getSupplierById(id) {
    try {
      const supplier = await prisma.suppliers.findUnique({
        where: { id }
      });

      return supplier;
    } catch (error) {
      console.error('Error in getSupplierById:', error);
      throw error;
    }
  }

  async createSupplier(supplierData) {
    try {
      const { name, contact_info } = supplierData;

      // Check if supplier name already exists
      const existingSupplier = await prisma.suppliers.findFirst({
        where: {
          name: {
            equals: name,
            mode: 'insensitive'
          }
        }
      });

      if (existingSupplier) {
        throw new Error('Supplier name already exists');
      }

      const supplier = await prisma.suppliers.create({
        data: {
          name,
          contact_info: contact_info || {}
        }
      });

      return supplier;
    } catch (error) {
      console.error('Error in createSupplier:', error);
      throw error;
    }
  }

  async updateSupplier(id, updateData) {
    try {
      const supplier = await prisma.suppliers.findUnique({
        where: { id }
      });

      if (!supplier) {
        return null;
      }

      // Check if new name conflicts with existing supplier
      if (updateData.name && updateData.name !== supplier.name) {
        const existingSupplier = await prisma.suppliers.findFirst({
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

        if (existingSupplier) {
          throw new Error('Supplier name already exists');
        }
      }

      const updatedSupplier = await prisma.suppliers.update({
        where: { id },
        data: updateData
      });

      return updatedSupplier;
    } catch (error) {
      console.error('Error in updateSupplier:', error);
      throw error;
    }
  }

  async deleteSupplier(id) {
    try {
      const supplier = await prisma.suppliers.findUnique({
        where: { id }
      });

      if (!supplier) {
        throw new Error('Supplier not found');
      }

      await prisma.suppliers.delete({
        where: { id }
      });
    } catch (error) {
      console.error('Error in deleteSupplier:', error);
      throw error;
    }
  }

  async getSupplierStats(id) {
    try {
      const supplier = await prisma.suppliers.findUnique({
        where: { id }
      });

      if (!supplier) {
        throw new Error('Supplier not found');
      }

      // Since there's no direct relation to stocks/medicines in the schema,
      // this would need to be extended based on your business logic
      // For now, returning basic supplier info
      return {
        supplier,
        stats: {
          totalOrders: 0, // Would need order tracking
          totalValue: 0,  // Would need order tracking
          lastOrderDate: null
        }
      };
    } catch (error) {
      console.error('Error in getSupplierStats:', error);
      throw error;
    }
  }
}

module.exports = new SuppliersService();
