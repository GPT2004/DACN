const express = require('express');
const router = express.Router();
const roomsController = require('./rooms.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { authorize } = require('../../middlewares/role.middleware');

const { createRoomSchema, updateRoomSchema, updateRoomStatusSchema } = require('./rooms.validator');

router.use(authenticate);

// Get all rooms
router.get('/', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  roomsController.getAllRooms
);

// Get deleted rooms (trash bin)
router.get('/deleted', 
  authorize(['Admin']), 
  roomsController.getDeletedRooms
);

// Get available rooms
router.get('/available', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  roomsController.getAvailableRooms
);

// Get room by ID
router.get('/:id', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  roomsController.getRoomById
);

// Get room schedule
router.get('/:id/schedule', 
  authorize(['Admin', 'Receptionist', 'Doctor']), 
  roomsController.getRoomSchedule
);

// Create room (Admin only)
router.post('/', 
  authorize(['Admin']), 
  validate(createRoomSchema), 
  roomsController.createRoom
);

// Update room (Admin only)
router.put('/:id', 
  authorize(['Admin']), 
  validate(updateRoomSchema), 
  roomsController.updateRoom
);

// Update room status (Admin, Receptionist)
router.patch('/:id/status', 
  authorize(['Admin', 'Receptionist']), 
  validate(updateRoomStatusSchema), 
  roomsController.updateRoomStatus
);

// Delete room (Admin only)
router.delete('/:id', 
  authorize(['Admin']), 
  roomsController.deleteRoom
);

// Restore room (Admin only)
router.patch('/:id/restore', 
  authorize(['Admin']), 
  roomsController.restoreRoom
);

module.exports = router;
