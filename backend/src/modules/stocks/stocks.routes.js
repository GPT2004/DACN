const express = require('express');
const router = express.Router();
const stocksController = require('./stocks.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const { 
  createStockSchema, 
  updateStockSchema, 
  adjustStockSchema 
} = require('./stocks.validator');

router.use(authenticate);

// Get all stocks
router.get('/', 
  authorize(['Admin', 'Pharmacist']), 
  stocksController.getAllStocks
);

// Get low stock items
router.get('/alerts/low-stock', 
  authorize(['Admin', 'Pharmacist']), 
  stocksController.getLowStockItems
);

// Get expiring stock
router.get('/alerts/expiring', 
  authorize(['Admin', 'Pharmacist']), 
  stocksController.getExpiringStock
);

// Get expired stock
router.get('/alerts/expired', 
  authorize(['Admin', 'Pharmacist']), 
  stocksController.getExpiredStock
);

// Get all stock alerts
router.get('/alerts', 
  authorize(['Admin', 'Pharmacist']), 
  stocksController.getStockAlerts
);

// Get stock by ID
router.get('/:id', 
  authorize(['Admin', 'Pharmacist']), 
  stocksController.getStockById
);

// Get medicine stock
router.get('/medicine/:medicineId', 
  authorize(['Admin', 'Pharmacist', 'Doctor']), 
  stocksController.getMedicineStock
);

// Get stock history
router.get('/medicine/:medicineId/history', 
  authorize(['Admin', 'Pharmacist']), 
  stocksController.getStockHistory
);

// Create stock
router.post('/', 
  authorize(['Admin', 'Pharmacist']), 
  validate(createStockSchema), 
  stocksController.createStock
);

// Update stock
router.put('/:id', 
  authorize(['Admin', 'Pharmacist']), 
  validate(updateStockSchema), 
  stocksController.updateStock
);

// Adjust stock quantity
router.patch('/:id/adjust', 
  authorize(['Admin', 'Pharmacist']), 
  validate(adjustStockSchema), 
  stocksController.adjustStockQuantity
);

// Delete stock
router.delete('/:id', 
  authorize(['Admin']), 
  stocksController.deleteStock
);

module.exports = router;
