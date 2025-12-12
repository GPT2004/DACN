const medicalRecordService = require('./medical_records.service');
const { successResponse } = require('../../utils/response');

class MedicalRecordController {
  async createMedicalRecord(req, res, next) {
    try {
      const record = await medicalRecordService.createMedicalRecord(req.validatedBody, req.user);
      return successResponse(res, record, 'Medical record created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async getMedicalRecords(req, res, next) {
    try {
      const { patient_id, doctor_id, page = 1, limit = 10 } = req.query;
      const result = await medicalRecordService.getMedicalRecords({
        patient_id,
        doctor_id,
        page: parseInt(page),
        limit: parseInt(limit),
      }, req.user);
      return successResponse(res, result, 'Medical records retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getMedicalRecordById(req, res, next) {
    try {
      const record = await medicalRecordService.getMedicalRecordById(req.params.id, req.user);
      return successResponse(res, record, 'Medical record retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateMedicalRecord(req, res, next) {
    try {
      const record = await medicalRecordService.updateMedicalRecord(
        req.params.id,
        req.validatedBody,
        req.user
      );
      return successResponse(res, record, 'Medical record updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteMedicalRecord(req, res, next) {
    try {
      await medicalRecordService.deleteMedicalRecord(req.params.id, req.user);
      return successResponse(res, null, 'Medical record deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async getPatientMedicalHistory(req, res, next) {
    try {
      const history = await medicalRecordService.getPatientMedicalHistory(
        req.params.patient_id,
        req.user
      );
      return successResponse(res, history, 'Patient medical history retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async uploadAttachment(req, res, next) {
    try {
      if (!req.file) {
        throw new Error('No file uploaded');
      }
      
      const record = await medicalRecordService.addAttachment(
        req.params.id,
        req.file,
        req.user
      );
      return successResponse(res, record, 'Attachment uploaded successfully');
    } catch (error) {
      next(error);
    }
  }

  async sendToPatient(req, res, next) {
    try {
      const result = await medicalRecordService.sendMedicalRecordToPatient(req.params.id, req.user);
      return successResponse(res, result, 'Medical record sent to patient');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = MedicalRecordController;