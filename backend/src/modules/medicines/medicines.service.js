const prisma = require('../../config/database');
const auditLogsService = require('../audit-logs/audit-logs.service');

class MedicineService {
  // Medicine CRUD
  async createMedicine(data) {
    // Check if code exists
    if (data.code) {
      const existing = await prisma.medicines.findUnique({
        where: { code: data.code },
      });

      if (existing) {
        throw new Error('Medicine code already exists');
      }
    }

    const medicine = await prisma.medicines.create({
      data: {
        name: data.name,
        code: data.code,
        description: data.description,
        unit: data.unit,
        price: data.price || 0,
      },
    });

    // Audit (fire-and-forget)
    try { auditLogsService.logAction(null, 'medicine.create', { medicine_id: medicine.id, name: medicine.name }); } catch (e) {}

    return medicine;
  }

  async getMedicines(filters) {
    const { search, page, limit } = filters;
    const skip = (page - 1) * limit;

    const where = {};

    // By default exclude soft-deleted records
    if (!filters || !filters.include_deleted) {
      where.deleted_at = null;
    }

    if (search) {
      where.OR = [
        { name: { contains: search, mode: 'insensitive' } },
        { code: { contains: search, mode: 'insensitive' } },
      ];
    }

    const [medicines, total] = await Promise.all([
      prisma.medicines.findMany({
        where,
        skip,
        take: limit,
        include: {
          stocks: {
            where: {
              expiry_date: { gt: new Date() },
            },
          },
        },
        orderBy: {
          name: 'asc',
        },
      }),
      prisma.medicines.count({ where }),
    ]);

    // Calculate total stock for each medicine
    const medicinesWithStock = medicines.map(medicine => {
      const totalStock = medicine.stocks.reduce((sum, stock) => sum + stock.quantity, 0);
      return {
        ...medicine,
        total_stock: totalStock,
      };
    });

    return {
      medicines: medicinesWithStock,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  async getMedicineById(id) {
    const medicine = await prisma.medicines.findUnique({
      where: { id: parseInt(id) },
      include: {
        stocks: {
          where: {
            expiry_date: { gt: new Date() },
          },
          orderBy: {
            expiry_date: 'asc',
          },
        },
      },
    });

    if (!medicine) {
      throw new Error('Medicine not found');
    }

    const totalStock = medicine.stocks.reduce((sum, stock) => sum + stock.quantity, 0);

    return {
      ...medicine,
      total_stock: totalStock,
    };
  }

  async updateMedicine(id, data) {
    const medicine = await prisma.medicines.findUnique({
      where: { id: parseInt(id) },
    });

    if (!medicine) {
      throw new Error('Medicine not found');
    }

    // Check code uniqueness if updating code
    if (data.code && data.code !== medicine.code) {
      const existing = await prisma.medicines.findUnique({
        where: { code: data.code },
      });

      if (existing) {
        throw new Error('Medicine code already exists');
      }
    }

    const updateData = {};
    if (data.name) updateData.name = data.name;
    if (data.code) updateData.code = data.code;
    if (data.description !== undefined) updateData.description = data.description;
    if (data.unit) updateData.unit = data.unit;
    if (data.price !== undefined) updateData.price = data.price;
    updateData.updated_at = new Date();

    const updated = await prisma.medicines.update({
      where: { id: parseInt(id) },
      data: updateData,
    });

    try { auditLogsService.logAction(null, 'medicine.update', { medicine_id: updated.id }); } catch (e) {}

    return updated;
  }

  async deleteMedicine(id) {
    const medicine = await prisma.medicines.findUnique({
      where: { id: parseInt(id) },
    });

    if (!medicine) {
      throw new Error('Medicine not found');
    }

    // Check if medicine is used in any prescriptions
    const usedInPrescriptions = await prisma.$queryRaw`
      SELECT COUNT(*) as count
      FROM prescriptions
      WHERE items::text LIKE '%"medicine_id":' || ${id} || '%'
    `;

    if (parseInt(usedInPrescriptions[0].count) > 0) {
      throw new Error('Cannot delete medicine that is used in prescriptions');
    }

    // Soft-delete: set deleted_at timestamp
    await prisma.medicines.update({
      where: { id: parseInt(id) },
      data: { deleted_at: new Date(), updated_at: new Date() }
    });

    // Audit log
    try {
      await auditLogsService.logAction(null, 'medicine.delete', { medicine_id: parseInt(id) });
    } catch (e) {
      // don't block operation if audit fails
    }

    return { message: 'Medicine soft-deleted successfully' };
  }

  async restoreMedicine(id) {
    const medicine = await prisma.medicines.findUnique({ where: { id: parseInt(id) } });
    if (!medicine) throw new Error('Medicine not found');

    await prisma.medicines.update({
      where: { id: parseInt(id) },
      data: { deleted_at: null, updated_at: new Date() }
    });

    try {
      await auditLogsService.logAction(null, 'medicine.restore', { medicine_id: parseInt(id) });
    } catch (e) {}

    return { message: 'Medicine restored successfully' };
  }

  // Stock CRUD
  async createStock(data) {
    // Verify medicine exists
    const medicine = await prisma.medicines.findUnique({
      where: { id: parseInt(data.medicine_id) },
    });

    if (!medicine) {
      throw new Error('Medicine not found');
    }

    const stock = await prisma.stocks.create({
      data: {
        medicine_id: parseInt(data.medicine_id),
        batch_number: data.batch_number,
        expiry_date: new Date(data.expiry_date),
        quantity: data.quantity,
      },
      include: {
        medicine: true,
      },
    });

    return stock;
  }

  async getStocks(filters) {
    const { medicine_id, low_stock, expiring_soon, page, limit } = filters;
    const skip = (page - 1) * limit;

    const where = {
      expiry_date: { gt: new Date() }, // Only non-expired stocks
    };

    if (medicine_id) {
      where.medicine_id = parseInt(medicine_id);
    }

    if (low_stock) {
      where.quantity = { lt: 20 }; // Low stock threshold
    }

    if (expiring_soon) {
      const thirtyDaysFromNow = new Date();
      thirtyDaysFromNow.setDate(thirtyDaysFromNow.getDate() + 30);
      where.expiry_date = {
        gt: new Date(),
        lte: thirtyDaysFromNow,
      };
    }

    const [stocks, total] = await Promise.all([
      prisma.stocks.findMany({
        where,
        skip,
        take: limit,
        include: {
          medicine: true,
        },
        orderBy: {
          expiry_date: 'asc',
        },
      }),
      prisma.stocks.count({ where }),
    ]);

    return {
      stocks,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  async getStockById(id) {
    const stock = await prisma.stocks.findUnique({
      where: { id: parseInt(id) },
      include: {
        medicine: true,
      },
    });

    if (!stock) {
      throw new Error('Stock not found');
    }

    return stock;
  }

  async updateStock(id, data) {
    const stock = await prisma.stocks.findUnique({
      where: { id: parseInt(id) },
    });

    if (!stock) {
      throw new Error('Stock not found');
    }

    const updateData = {};
    if (data.batch_number) updateData.batch_number = data.batch_number;
    if (data.expiry_date) updateData.expiry_date = new Date(data.expiry_date);
    if (data.quantity !== undefined) updateData.quantity = data.quantity;
    updateData.updated_at = new Date();

    const updated = await prisma.stocks.update({
      where: { id: parseInt(id) },
      data: updateData,
      include: {
        medicine: true,
      },
    });

    return updated;
  }

  async deleteStock(id) {
    const stock = await prisma.stocks.findUnique({
      where: { id: parseInt(id) },
    });

    if (!stock) {
      throw new Error('Stock not found');
    }

    await prisma.stocks.delete({
      where: { id: parseInt(id) },
    });

    return { message: 'Stock deleted successfully' };
  }

  // Reports and Alerts
  async getStockSummary() {
    const totalMedicines = await prisma.medicines.count();

    const totalStockValue = await prisma.$queryRaw`
      SELECT COALESCE(SUM(s.quantity * m.price), 0)::INTEGER as total_value
      FROM stocks s
      JOIN medicines m ON s.medicine_id = m.id
      WHERE s.expiry_date > NOW()
    `;

    const lowStockCount = await prisma.$queryRaw`
      SELECT COUNT(DISTINCT m.id)::INTEGER as count
      FROM medicines m
      JOIN stocks s ON m.id = s.medicine_id
      WHERE s.expiry_date > NOW()
      GROUP BY m.id
      HAVING SUM(s.quantity) < 20
    `;

    const expiringCount = await prisma.stocks.count({
      where: {
        expiry_date: {
          gt: new Date(),
          lte: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
        },
      },
    });

    return {
      total_medicines: totalMedicines,
      total_stock_value: totalStockValue[0].total_value || 0,
      low_stock_items: lowStockCount[0]?.count || 0,
      expiring_soon: expiringCount,
    };
  }

  async getLowStockAlerts() {
    const lowStockMedicines = await prisma.$queryRaw`
      SELECT 
        m.id,
        m.name,
        m.code,
        m.unit,
        COALESCE(SUM(s.quantity), 0)::INTEGER as total_quantity
      FROM medicines m
      LEFT JOIN stocks s ON m.id = s.medicine_id AND s.expiry_date > NOW()
      GROUP BY m.id, m.name, m.code, m.unit
      HAVING COALESCE(SUM(s.quantity), 0) < 20
      ORDER BY total_quantity ASC
    `;

    return lowStockMedicines;
  }

  async getExpiringMedicines(days = 30) {
    const futureDate = new Date();
    futureDate.setDate(futureDate.getDate() + days);

    const expiringStocks = await prisma.stocks.findMany({
      where: {
        expiry_date: {
          gt: new Date(),
          lte: futureDate,
        },
        quantity: { gt: 0 },
      },
      include: {
        medicine: true,
      },
      orderBy: {
        expiry_date: 'asc',
      },
    });

    return expiringStocks;
  }

  async importMedicines(medicinesData) {
    const results = {
      success: [],
      errors: [],
      total: medicinesData.length
    };
    
    for (const medicineData of medicinesData) {
      try {
        // Check if medicine already exists by code or name
        let medicine;
        
        if (medicineData.code) {
          medicine = await prisma.medicines.findUnique({
            where: { code: medicineData.code }
          });
        }
        
        if (!medicine) {
          // Try to find by name
          medicine = await prisma.medicines.findFirst({
            where: { name: medicineData.name }
          });
        }
        
        if (!medicine) {
          // Create new medicine
          medicine = await prisma.medicines.create({
            data: {
              name: medicineData.name,
              code: medicineData.code,
              description: medicineData.description,
              unit: medicineData.unit,
              price: medicineData.price || 0,
            },
          });
        }
        
        // Create stock entry
        const stock = await prisma.stocks.create({
          data: {
            medicine_id: medicine.id,
            batch_number: medicineData.batch_number,
            expiry_date: new Date(medicineData.expiry_date),
            quantity: medicineData.quantity,
          },
          include: {
            medicine: true,
          },
        });
        
        results.success.push({
          medicine: medicine,
          stock: stock,
        });
        
      } catch (error) {
        results.errors.push({
          data: medicineData,
          error: error.message,
        });
      }
    }
    // Audit import summary
    try { auditLogsService.logAction(null, 'medicine.import', { total: results.total, success: results.success.length, errors: results.errors.length }); } catch (e) {}

    return results;
  }
}

module.exports = new MedicineService();
