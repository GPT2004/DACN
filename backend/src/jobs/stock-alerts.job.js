const prisma = require('../config/database');
const logger = require('../utils/logger');
const { STOCK_ALERTS } = require('../config/constants');

class StockAlertsJob {
  async checkLowStock() {
    try {
      const lowStockMedicines = await prisma.$queryRaw`
        SELECT 
          m.id, 
          m.name, 
          SUM(s.quantity) as total_quantity
        FROM medicines m
        LEFT JOIN stocks s ON m.id = s.medicine_id AND s.expiry_date > CURRENT_DATE
        GROUP BY m.id, m.name
        HAVING COALESCE(SUM(s.quantity), 0) < ${STOCK_ALERTS.LOW_STOCK_THRESHOLD}
      `;

      if (lowStockMedicines.length === 0) {
        logger.info('No low stock medicines found');
        return;
      }

      logger.info(`Found ${lowStockMedicines.length} low stock medicines`);

      // Get all admins
      const admins = await prisma.users.findMany({
        where: {
          role: { name: 'Admin' },
          is_active: true,
        },
      });

      // Send notifications to admins
      const notifications = admins.map(admin => ({
        user_id: admin.id,
        type: 'LOW_STOCK',
        payload: {
          message: `${lowStockMedicines.length} loại thuốc sắp hết`,
          medicines: lowStockMedicines.map(m => ({
            id: m.id,
            name: m.name,
            quantity: parseInt(m.total_quantity),
          })),
        },
      }));

      await prisma.notifications.createMany({ data: notifications });

      logger.info(`Sent ${notifications.length} low stock alerts`);
    } catch (error) {
      logger.error('Low stock alert job error:', error);
    }
  }

  async checkExpiringMedicines() {
    try {
      const expiryDate = new Date();
      expiryDate.setDate(expiryDate.getDate() + STOCK_ALERTS.EXPIRY_WARNING_DAYS);

      const expiringStocks = await prisma.stocks.findMany({
        where: {
          expiry_date: {
            lte: expiryDate,
            gt: new Date(),
          },
          quantity: {
            gt: 0,
          },
        },
        include: {
          medicine: true,
        },
      });

      if (expiringStocks.length === 0) {
        logger.info('No expiring medicines found');
        return;
      }

      logger.info(`Found ${expiringStocks.length} expiring medicine batches`);

      // Get all admins
      const admins = await prisma.users.findMany({
        where: {
          role: { name: 'Admin' },
          is_active: true,
        },
      });

      // Send notifications
      const notifications = admins.map(admin => ({
        user_id: admin.id,
        type: 'EXPIRING_MEDICINES',
        payload: {
          message: `${expiringStocks.length} lô thuốc sắp hết hạn`,
          stocks: expiringStocks.map(s => ({
            medicine_name: s.medicine.name,
            batch_number: s.batch_number,
            expiry_date: s.expiry_date,
            quantity: s.quantity,
          })),
        },
      }));

      await prisma.notifications.createMany({ data: notifications });

      logger.info(`Sent ${notifications.length} expiring medicines alerts`);
    } catch (error) {
      logger.error('Expiring medicines alert job error:', error);
    }
  }
}

module.exports = new StockAlertsJob();