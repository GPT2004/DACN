const express = require('express');
const router = express.Router();
const authController = require('./auth.controller');
const { validate } = require('../../middlewares/validate.middleware');
const { authenticate } = require('../../middlewares/auth.middleware');
const { registerSchema, loginSchema, changePasswordSchema } = require('./auth.validator');

router.post('/register', validate(registerSchema), authController.register);
router.get('/confirm-registration', authController.confirmRegistration);
router.post('/login', validate(loginSchema), authController.login);
router.post('/forgot-password', authController.forgotPassword);
router.get('/confirm-reset', authController.confirmReset);
router.get('/check-email', authController.checkEmail);
router.post('/logout', authenticate, authController.logout);
router.get('/me', authenticate, authController.getProfile);
router.put('/change-password', authenticate, validate(changePasswordSchema), authController.changePassword);

module.exports = router;