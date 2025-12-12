const patientsService = require('./patients.service');
const { successResponse } = require('../../utils/response');
const { uploadBuffer } = require('../../utils/cloudinary');

class PatientsController {
  async getAllPatients(req, res, next) {
    try {
      const { page = 1, limit = 10, search, gender, blood_type } = req.query;

      const filters = {};
      if (search) filters.search = search;
      if (gender) filters.gender = gender;
      if (blood_type) filters.blood_type = blood_type;

      const result = await patientsService.getAllPatients(filters, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Patients retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getPatientById(req, res, next) {
    try {
      const { id } = req.params;
      const userId = req.user.id;
      const userRole = req.user.role.name;

      // If patient, check if they can only view their own profile
      if (userRole === 'Patient') {
        const patient = await patientsService.getPatientByUserId(userId);
        if (patient.id !== parseInt(id)) {
          throw new Error('You can only view your own profile');
        }
      }

      const patient = await patientsService.getPatientById(parseInt(id));

      if (!patient) {
        throw new Error('Patient not found');
      }

      return successResponse(res, patient, 'Patient retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getPatientMedicalRecords(req, res, next) {
    try {
      const { id } = req.params;
      const { page = 1, limit = 20 } = req.query;

      const result = await patientsService.getPatientMedicalRecords(
        parseInt(id),
        { page: parseInt(page), limit: parseInt(limit) }
      );

      return successResponse(res, result, 'Medical records retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getPatientAppointments(req, res, next) {
    try {
      const { id } = req.params;
      const { status, startDate, endDate, page = 1, limit = 20 } = req.query;

      const filters = {};
      if (status) filters.status = status;
      if (startDate) filters.startDate = startDate;
      if (endDate) filters.endDate = endDate;

      const result = await patientsService.getPatientAppointments(
        parseInt(id),
        filters,
        { page: parseInt(page), limit: parseInt(limit) }
      );

      return successResponse(res, result, 'Appointments retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getPatientPrescriptions(req, res, next) {
    try {
      const { id } = req.params;
      const { page = 1, limit = 20 } = req.query;

      const result = await patientsService.getPatientPrescriptions(
        parseInt(id),
        { page: parseInt(page), limit: parseInt(limit) }
      );

      return successResponse(res, result, 'Prescriptions retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getPatientInvoices(req, res, next) {
    try {
      const { id } = req.params;
      const { status, page = 1, limit = 20 } = req.query;

      const filters = {};
      if (status) filters.status = status;

      const result = await patientsService.getPatientInvoices(
        parseInt(id),
        filters,
        { page: parseInt(page), limit: parseInt(limit) }
      );

      return successResponse(res, result, 'Invoices retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getMyProfile(req, res, next) {
    try {
      const userId = req.user.id;
      // Ensure a patient row exists for this user; create a minimal one if missing
      const patient = await patientsService.ensurePatientForUser(userId);

      if (!patient) {
        const err = new Error('Patient profile not found');
        err.statusCode = 404;
        throw err;
      }

      return successResponse(res, patient, 'Profile retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getMyPatients(req, res, next) {
    try {
      const userId = req.user.id;
      const result = await patientsService.getPatientsByOwner(userId);
      return successResponse(res, result, 'My patients retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getMySharedMedicalRecords(req, res, next) {
    try {
      const userId = req.user.id;
      const rows = await patientsService.getSharedMedicalRecordsForUser(userId);
      return successResponse(res, rows, 'Shared medical records for user retrieved');
    } catch (error) {
      next(error);
    }
  }

  async createDependentPatient(req, res, next) {
    try {
      const userId = req.user.id;
      const patientData = req.validatedBody;
      const newPatient = await patientsService.createDependentPatient(userId, patientData);
      return successResponse(res, newPatient, 'Dependent patient created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async updateDependentProfile(req, res, next) {
    try {
      const ownerId = req.user.id;
      const { id } = req.params;
      const updateData = req.validatedBody;

      if (req.file) {
        // Support both memory and disk storage from multer
        const fs = require('fs').promises;
        let buffer;
        if (req.file.buffer) {
          buffer = req.file.buffer;
        } else if (req.file.path) {
          buffer = await fs.readFile(req.file.path);
        }

        if (buffer) {
          const options = {
            folder: process.env.CLOUDINARY_AVATAR_FOLDER || 'avatars',
            resource_type: 'image',
            public_id: `patient_${id}_${Date.now()}`
          };
          const result = await uploadBuffer(buffer, options);
          updateData.avatar_url = result.secure_url;

          // remove local file if present
          if (req.file.path) {
            try { await fs.unlink(req.file.path); } catch (e) {}
          }
        }
      }

      const updatedPatient = await patientsService.updateDependentPatient(ownerId, parseInt(id), updateData);

      if (!updatedPatient) {
        throw new Error('Patient not found or not authorized');
      }

      return successResponse(res, updatedPatient, 'Dependent profile updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteMyDependent(req, res, next) {
    try {
      const ownerId = req.user.id;
      const { id } = req.params;
      const patient = await patientsService.getPatientById(parseInt(id));
      if (!patient) {
        throw new Error('Patient not found');
      }

      // Only owner or linked user can delete
      if (patient.owner_user_id !== ownerId && patient.user_id !== ownerId) {
        const err = new Error('Not authorized to delete this patient');
        err.status = 403;
        throw err;
      }

      await patientsService.deletePatient(parseInt(id));
      return successResponse(res, null, 'Dependent patient deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteMyProfile(req, res, next) {
    try {
      const userId = req.user.id;
      const patient = await patientsService.getPatientByUserId(userId);
      if (!patient) {
        throw new Error('Patient profile not found');
      }

      await patientsService.deletePatient(patient.id);

      // Optionally you may want to revoke tokens / logout the user on the client side.
      return successResponse(res, null, 'Your profile has been deleted');
    } catch (error) {
      next(error);
    }
  }

  async updateMyProfile(req, res, next) {
    try {
      const userId = req.user.id;
      const updateData = req.validatedBody;

      // Nếu có file upload, upload lên Cloudinary và thêm đường dẫn vào updateData
      if (req.file) {
        const fs = require('fs').promises;
        let buffer;
        if (req.file.buffer) {
          buffer = req.file.buffer;
        } else if (req.file.path) {
          buffer = await fs.readFile(req.file.path);
        }

        if (buffer) {
          const options = {
            folder: process.env.CLOUDINARY_AVATAR_FOLDER || 'avatars',
            resource_type: 'image',
            public_id: `patient_user_${userId}_${Date.now()}`
          };
          const result = await uploadBuffer(buffer, options);
          updateData.avatar_url = result.secure_url;

          if (req.file.path) {
            try { await fs.unlink(req.file.path); } catch (e) {}
          }
        }
      }

      const updatedPatient = await patientsService.updatePatientProfile(
        userId,
        updateData
      );

      if (!updatedPatient) {
        throw new Error('Patient profile not found');
      }

      return successResponse(res, updatedPatient, 'Profile updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async createPatient(req, res, next) {
    try {
      const patientData = req.validatedBody;
      const newPatient = await patientsService.createPatient(patientData);

      return successResponse(res, newPatient, 'Patient created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async createPatientQuick(req, res, next) {
    try {
      const patientData = req.validatedBody;
      const newPatient = await patientsService.createPatientWithoutUser(patientData);
      return successResponse(res, newPatient, 'Patient (quick) created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async updatePatient(req, res, next) {
    try {
      const { id } = req.params;
      const updateData = req.validatedBody;

      const updatedPatient = await patientsService.updatePatient(
        parseInt(id),
        updateData
      );

      if (!updatedPatient) {
        throw new Error('Patient not found');
      }

      return successResponse(res, updatedPatient, 'Patient updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deletePatient(req, res, next) {
    try {
      const { id } = req.params;
      await patientsService.deletePatient(parseInt(id));

      return successResponse(res, null, 'Patient deleted successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new PatientsController();