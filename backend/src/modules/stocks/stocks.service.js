/* eslint-disable no-console */
const prisma = require('../../config/database');

class StocksService {
  async getAllStocks(filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {};

      if (filters.medicine_id) {
        where.medicine_id = filters.medicine_id;
      }

      // Filter for valid stock (not expired)
      if (!filters.expiring) {
        where.expiry_date = {
          gte: new Date()
        };
      }

      const [total, stocks] = await Promise.all([
        prisma.stocks.count({ where }),
        prisma.stocks.findMany({
          where,
          skip,
          take: limit,
          include: {
            medicine: {
              select: {
                id: true,
                name: true,
                code: true,
                unit: true,
                price: true
              }
            }
          },
          orderBy: {
            expiry_date: 'asc'
          }
        })
      ]);

      return {
        stocks,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getAllStocks:', error);
      throw error;
    }
  }

  async getStockById(id) {
    try {
      const stock = await prisma.stocks.findUnique({
        where: { id },
        include: {
          medicine: true
        }
      });

      return stock;
    } catch (error) {
      console.error('Error in getStockById:', error);
      throw error;
    }
  }

  async getmedicinetock(medicineId) {
    try {
      const stocks = await prisma.stocks.findMany({
        where: {
          medicine_id: medicineId,
          expiry_date: {
            gte: new Date()
          }
        },
        orderBy: {
          expiry_date: 'asc'
        }
      });

      const totalQuantity = stocks.reduce((sum, stock) => sum + stock.quantity, 0);

      const medicine = await prisma.medicine.findUnique({
        where: { id: medicineId }
      });

      return {
        medicine,
        stocks,
        totalQuantity
      };
    } catch (error) {
      console.error('Error in getmedicinetock:', error);
      throw error;
    }
  }

  async getLowStockItems(threshold = 20) {
    try {
      const lowStockItems = await prisma.$queryRaw`
        SELECT 
          m.id,
          m.name,
          m.code,
          m.unit,
          COALESCE(SUM(s.quantity), 0) as total_quantity
        FROM medicine m
        LEFT JOIN stocks s ON m.id = s.medicine_id 
          AND (s.expiry_date > CURRENT_DATE OR s.expiry_date IS NULL)
        GROUP BY m.id, m.name, m.code, m.unit
        HAVING COALESCE(SUM(s.quantity), 0) < ${threshold}
        ORDER BY total_quantity ASC
      `;

      return lowStockItems;
    } catch (error) {
      console.error('Error in getLowStockItems:', error);
      throw error;
    }
  }

  async getExpiringStock(days = 30) {
    try {
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + days);

      const expiringStock = await prisma.stocks.findMany({
        where: {
          expiry_date: {
            gte: new Date(),
            lte: futureDate
          },
          quantity: {
            gt: 0
          }
        },
        include: {
          medicine: {
            select: {
              name: true,
              code: true,
              unit: true
            }
          }
        },
        orderBy: {
          expiry_date: 'asc'
        }
      });

      return expiringStock;
    } catch (error) {
      console.error('Error in getExpiringStock:', error);
      throw error;
    }
  }

  async getExpiredStock() {
    try {
      const expiredStock = await prisma.stocks.findMany({
        where: {
          expiry_date: {
            lt: new Date()
          },
          quantity: {
            gt: 0
          }
        },
        include: {
          medicine: {
            select: {
              name: true,
              code: true,
              unit: true
            }
          }
        },
        orderBy: {
          expiry_date: 'desc'
        }
      });

      return expiredStock;
    } catch (error) {
      console.error('Error in getExpiredStock:', error);
      throw error;
    }
  }

  async createStock(stockData) {
  try {
    const { medicine_id, batch_number, expiry_date, quantity } = stockData;

    // KIỂM TRA PRISMA
    if (!prisma || !prisma.stocks) {
      throw new Error('Prisma client not initialized');
    }

    // Kiểm tra thuốc tồn tại
    const medicine = await prisma.medicines.findUnique({
      where: { id: medicine_id }
    });

    if (!medicine) {
      throw new Error('Medicine not found');
    }

    // Kiểm tra batch trùng
    const existingStock = await prisma.stocks.findUnique({
      where: {
        medicine_id_batch_number: {
          medicine_id,
          batch_number
        }
      }
    });

    if (existingStock) {
      throw new Error('Batch number already exists for this medicine');
    }

    const stock = await prisma.stocks.create({
      data: {
        medicine_id,
        batch_number,
        expiry_date: new Date(expiry_date),
        quantity
      },
      include: {
        medicine: true  // ĐÃ SỬA TỪ `medicines` → `medicine`
      }
    });

    return stock;
  } catch (error) {
    console.error('Error in createStock:', error);
    throw error;
  }
}

  async updateStock(id, updateData) {
    try {
      const stock = await prisma.stocks.findUnique({
        where: { id }
      });

      if (!stock) {
        return null;
      }

      const updatedStock = await prisma.stocks.update({
        where: { id },
        data: {
          ...updateData,
          expiry_date: updateData.expiry_date ? new Date(updateData.expiry_date) : undefined,
          updated_at: new Date()
        },
        include: {
          medicine: true
        }
      });

      return updatedStock;
    } catch (error) {
      console.error('Error in updateStock:', error);
      throw error;
    }
  }

  async adjustStockQuantity(stockId, adjustment, reason, userId) {
    try {
      const stock = await prisma.stocks.findUnique({
        where: { id: stockId }
      });

      if (!stock) {
        throw new Error('Stock not found');
      }

      const newQuantity = stock.quantity + adjustment;

      if (newQuantity < 0) {
        throw new Error('Insufficient stock quantity');
      }

      const updatedStock = await prisma.$transaction(async (tx) => {
        // Update stock quantity
        const updated = await tx.stocks.update({
          where: { id: stockId },
          data: {
            quantity: newQuantity,
            updated_at: new Date()
          },
          include: {
            medicine: true
          }
        });

        // Log the adjustment in audit logs
        await tx.audit_logs.create({
          data: {
            user_id: userId,
            action: `STOCK_ADJUSTMENT: ${adjustment > 0 ? 'ADDED' : 'REMOVED'} ${Math.abs(adjustment)} units`,
            meta: {
              stock_id: stockId,
              medicine_id: stock.medicine_id,
              old_quantity: stock.quantity,
              new_quantity: newQuantity,
              adjustment,
              reason
            }
          }
        });

        return updated;
      });

      return updatedStock;
    } catch (error) {
      console.error('Error in adjustStockQuantity:', error);
      throw error;
    }
  }

  async deleteStock(id) {
    try {
      const stock = await prisma.stocks.findUnique({
        where: { id }
      });

      if (!stock) {
        throw new Error('Stock not found');
      }

      if (stock.quantity > 0) {
        throw new Error('Cannot delete stock with remaining quantity');
      }

      await prisma.stocks.delete({
        where: { id }
      });
    } catch (error) {
      console.error('Error in deleteStock:', error);
      throw error;
    }
  }

  async getStockHistory(medicineId, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const [total, history] = await Promise.all([
        prisma.stocks.count({ where: { medicine_id: medicineId } }),
        prisma.stocks.findMany({
          where: { medicine_id: medicineId },
          skip,
          take: limit,
          orderBy: {
            created_at: 'desc'
          }
        })
      ]);

      return {
        history,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getStockHistory:', error);
      throw error;
    }
  }

  async getStockAlerts() {
    try {
      const [lowStock, expiring, expired] = await Promise.all([
        this.getLowStockItems(20),
        this.getExpiringStock(30),
        this.getExpiredStock()
      ]);

      return {
        lowStock: {
          count: lowStock.length,
          items: lowStock
        },
        expiring: {
          count: expiring.length,
          items: expiring
        },
        expired: {
          count: expired.length,
          items: expired
        }
      };
    } catch (error) {
      console.error('Error in getStockAlerts:', error);
      throw error;
    }
  }

  async deductStock(medicineId, quantity) {
    try {
      // Get available stocks (not expired, ordered by expiry date - FIFO)
      const stocks = await prisma.stocks.findMany({
        where: {
          medicine_id: medicineId,
          expiry_date: {
            gte: new Date()
          },
          quantity: {
            gt: 0
          }
        },
        orderBy: {
          expiry_date: 'asc'
        }
      });

      const totalAvailable = stocks.reduce((sum, s) => sum + s.quantity, 0);

      if (totalAvailable < quantity) {
        throw new Error('Insufficient stock available');
      }

      let remaining = quantity;
      const updates = [];

      for (const stock of stocks) {
        if (remaining <= 0) break;

        const deduct = Math.min(stock.quantity, remaining);
        
        updates.push(
          prisma.stocks.update({
            where: { id: stock.id },
            data: {
              quantity: stock.quantity - deduct,
              updated_at: new Date()
            }
          })
        );

        remaining -= deduct;
      }

      await prisma.$transaction(updates);

      return {
        deducted: quantity,
        remaining: totalAvailable - quantity
      };
    } catch (error) {
      console.error('Error in deductStock:', error);
      throw error;
    }
  }
}

module.exports = new StocksService();
