const express = require('express');
const router = express.Router();
const suppliersController = require('./suppliers.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const { 
  createSupplierSchema, 
  updateSupplierSchema 
} = require('./suppliers.validator');

router.use(authenticate);

// Get all suppliers
router.get('/', 
  authorize(['Admin', 'Pharmacist']), 
  suppliersController.getAllSuppliers
);

// Get supplier by ID
router.get('/:id', 
  authorize(['Admin', 'Pharmacist']), 
  suppliersController.getSupplierById
);

// Get supplier stats
router.get('/:id/stats', 
  authorize(['Admin']), 
  suppliersController.getSupplierStats
);

// Create supplier
router.post('/', 
  authorize(['Admin']), 
  validate(createSupplierSchema), 
  suppliersController.createSupplier
);

// Update supplier
router.put('/:id', 
  authorize(['Admin']), 
  validate(updateSupplierSchema), 
  suppliersController.updateSupplier
);

// Delete supplier
router.delete('/:id', 
  authorize(['Admin']), 
  suppliersController.deleteSupplier
);

module.exports = router;