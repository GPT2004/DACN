const express = require('express');
const router = express.Router();
const specialtiesController = require('./specialties.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const { createSpecialtySchema, updateSpecialtySchema } = require('./specialties.validator');
const { uploadSingle } = require('../../middlewares/upload.middleware');

router.use(authenticate);

// Get all specialties
router.get('/',
  authorize(['Admin', 'Receptionist', 'Doctor']),
  specialtiesController.getAllSpecialties
);

// Get deleted specialties (trash bin)
router.get('/deleted',
  authorize(['Admin']),
  specialtiesController.getDeletedSpecialties
);

// Get specialty by ID
router.get('/:id',
  authorize(['Admin', 'Receptionist', 'Doctor']),
  specialtiesController.getSpecialtyById
);

// Create specialty (Admin only)
router.post('/',
  authorize(['Admin']),
  uploadSingle('image_url'),
  validate(createSpecialtySchema),
  specialtiesController.createSpecialty
);

// Update specialty (Admin only)
router.put('/:id',
  authorize(['Admin']),
  uploadSingle('image_url'),
  validate(updateSpecialtySchema),
  specialtiesController.updateSpecialty
);

// Delete specialty (Admin only)
router.delete('/:id',
  authorize(['Admin']),
  specialtiesController.deleteSpecialty
);

// Restore specialty (Admin only)
router.patch('/:id/restore',
  authorize(['Admin']),
  specialtiesController.restoreSpecialty
);

module.exports = router;