const appointmentService = require('./appointments.service');
const { successResponse } = require('../../utils/response');

class AppointmentController {
  async createAppointment(req, res, next) {
    try {
      const appointment = await appointmentService.createAppointment(req.validatedBody, req.user);
      return successResponse(res, appointment, 'Appointment created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async getAppointments(req, res, next) {
    try {
      const { status, date, doctor_id, patient_id, page = 1, limit = 10 } = req.query;
      const result = await appointmentService.getAppointments({
        status,
        date,
        doctor_id,
        patient_id,
        page: parseInt(page),
        limit: parseInt(limit),
      }, req.user);
      return successResponse(res, result, 'Appointments retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getAppointmentById(req, res, next) {
    try {
      const appointment = await appointmentService.getAppointmentById(req.params.id, req.user);
      return successResponse(res, appointment, 'Appointment retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateAppointment(req, res, next) {
    try {
      const appointment = await appointmentService.updateAppointment(
        req.params.id,
        req.validatedBody,
        req.user
      );
      return successResponse(res, appointment, 'Appointment updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async cancelAppointment(req, res, next) {
    try {
      const { reason } = req.body;
      const appointment = await appointmentService.cancelAppointment(req.params.id, reason, req.user);
      return successResponse(res, appointment, 'Appointment cancelled successfully');
    } catch (error) {
      next(error);
    }
  }

  async confirmAppointment(req, res, next) {
    try {
      const appointment = await appointmentService.confirmAppointment(req.params.id);
      return successResponse(res, appointment, 'Appointment confirmed successfully');
    } catch (error) {
      next(error);
    }
  }

  async checkInAppointment(req, res, next) {
    try {
      const appointment = await appointmentService.checkInAppointment(req.params.id);
      return successResponse(res, appointment, 'Patient checked in successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteAppointment(req, res, next) {
    try {
      const result = await appointmentService.deleteAppointment(req.params.id, req.user);
      return successResponse(res, result, 'Appointment deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async startAppointment(req, res, next) {
    try {
      const appointment = await appointmentService.startAppointment(req.params.id);
      return successResponse(res, appointment, 'Appointment started successfully');
    } catch (error) {
      next(error);
    }
  }

  async completeAppointment(req, res, next) {
    try {
      const appointment = await appointmentService.completeAppointment(req.params.id);
      return successResponse(res, appointment, 'Appointment completed successfully');
    } catch (error) {
      next(error);
    }
  }

  async markNoShow(req, res, next) {
    try {
      const appointment = await appointmentService.markNoShow(req.params.id);
      return successResponse(res, appointment, 'Appointment marked as no-show');
    } catch (error) {
      next(error);
    }
  }

  async rescheduleAppointment(req, res, next) {
    try {
      const { id } = req.params;
      const { timeslotId } = req.validatedBody;
      const appointment = await appointmentService.rescheduleAppointment(id, timeslotId, req.user);
      return successResponse(res, appointment, 'Appointment rescheduled successfully');
    } catch (error) {
      next(error);
    }
  }

  async getMyAppointments(req, res, next) {
    try {
      const { page = 1, limit = 10 } = req.query;
      const result = await appointmentService.getMyAppointmentsForPatient(req.user, {
        page: parseInt(page),
        limit: parseInt(limit),
      });
      return successResponse(res, result, 'My appointments (upcoming + history) retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getAppointmentHistory(req, res, next) {
    try {
      const result = await appointmentService.getAppointmentHistory(req.user);
      return successResponse(res, result, 'Appointment history retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getAllAppointmentsByPatient(req, res, next) {
    try {
      const result = await appointmentService.getAllAppointmentsByPatient(req.user);
      return successResponse(res, result, 'All appointments retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async patientConfirmAppointment(req, res, next) {
    try {
      const { id } = req.params;
      const result = await appointmentService.patientConfirmAppointment(id, req.user);
      return successResponse(res, result, 'Appointment confirmed by patient and reception notified');
    } catch (error) {
      next(error);
    }
  }

  // Public confirmation via token (email link)
  async confirmByToken(req, res, next) {
    try {
      const { id } = req.params;
      const { token } = req.query;
      const result = await appointmentService.confirmByToken(id, token);
      return successResponse(res, result, 'Appointment confirmed via email');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new AppointmentController();