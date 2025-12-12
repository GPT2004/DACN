const authService = require('./auth.service');
const { successResponse } = require('../../utils/response');

class AuthController {
  async register(req, res, next) {
    try {
      const result = await authService.register(req.validatedBody);
      return successResponse(res, result, 'Registration successful', 201);
    } catch (error) {
      next(error);
    }
  }

  async confirmRegistration(req, res, next) {
    try {
      const { token } = req.query;
      const result = await authService.confirmRegistration(token);
      return successResponse(res, result, 'Registration confirmed');
    } catch (error) {
      next(error);
    }
  }

  async forgotPassword(req, res, next) {
    try {
      const { email } = req.body;
      const result = await authService.forgotPassword(email);
      return successResponse(res, result, 'If the email exists, a confirmation was sent');
    } catch (error) {
      next(error);
    }
  }

  async confirmReset(req, res, next) {
    try {
      const { token } = req.query;
      const result = await authService.confirmReset(token);
      return successResponse(res, result, 'Password reset successful');
    } catch (error) {
      next(error);
    }
  }

  async checkEmail(req, res, next) {
    try {
      const { email } = req.query;
      const exists = await authService.checkEmail(email);
      return successResponse(res, { exists }, 'Email check');
    } catch (error) {
      next(error);
    }
  }

  async login(req, res, next) {
    try {
      const { email, password } = req.validatedBody;
      const result = await authService.login(email, password);
      return successResponse(res, result, 'Login successful');
    } catch (error) {
      next(error);
    }
  }

  async getProfile(req, res, next) {
    try {
      const profile = await authService.getProfile(req.user.id);
      return successResponse(res, profile, 'Profile retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async changePassword(req, res, next) {
    try {
      const { current_password, new_password } = req.validatedBody;
      const result = await authService.changePassword(
        req.user.id,
        current_password,
        new_password
      );
      return successResponse(res, result, 'Password changed successfully');
    } catch (error) {
      next(error);
    }
  }

  async logout(req, res, next) {
    try {
      // In a production app, you might want to blacklist the token
      return successResponse(res, null, 'Logout successful');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new AuthController();