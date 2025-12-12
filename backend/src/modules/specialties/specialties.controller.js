const specialtiesService = require('./specialties.service');
const { successResponse } = require('../../utils/response');
const { uploadBuffer } = require('../../utils/cloudinary');

class SpecialtiesController {
  async getAllSpecialties(req, res, next) {
    try {
      const { page = 1, limit = 20, name } = req.query;

      const filters = {};
      if (name) filters.name = name;

      const result = await specialtiesService.getAllSpecialties(filters, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Specialties retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getSpecialtyById(req, res, next) {
    try {
      const { id } = req.params;
      const specialty = await specialtiesService.getSpecialtyById(parseInt(id));

      if (!specialty) {
        throw new Error('Specialty not found');
      }

      return successResponse(res, specialty, 'Specialty retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async createSpecialty(req, res, next) {
    try {
      const specialtyData = req.validatedBody;

      // If a file was uploaded in field `image_url`, upload to Cloudinary and set image_url
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
            folder: process.env.CLOUDINARY_SPECIALTY_FOLDER || 'specialties',
            resource_type: 'image',
            public_id: `specialty_${Date.now()}`
          };
          const result = await uploadBuffer(buffer, options);
          specialtyData.image_url = result.secure_url;

          // cleanup local file
          if (req.file.path) {
            try { await fs.unlink(req.file.path); } catch (e) {}
          }
        }
      }

      const newSpecialty = await specialtiesService.createSpecialty(specialtyData);

      return successResponse(res, newSpecialty, 'Specialty created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async updateSpecialty(req, res, next) {
    try {
      const { id } = req.params;
      const updateData = req.validatedBody;

      // If file uploaded for image_url, upload and set updateData.image_url
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
            folder: process.env.CLOUDINARY_SPECIALTY_FOLDER || 'specialties',
            resource_type: 'image',
            public_id: `specialty_${id}_${Date.now()}`
          };
          const result = await uploadBuffer(buffer, options);
          updateData.image_url = result.secure_url;

          if (req.file.path) {
            try { await fs.unlink(req.file.path); } catch (e) {}
          }
        }
      }

      const updatedSpecialty = await specialtiesService.updateSpecialty(parseInt(id), updateData);

      if (!updatedSpecialty) {
        throw new Error('Specialty not found');
      }

      return successResponse(res, updatedSpecialty, 'Specialty updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteSpecialty(req, res, next) {
    try {
      const { id } = req.params;
      const userId = req.user.id;

      await specialtiesService.deleteSpecialty(parseInt(id), userId);

      return successResponse(res, null, 'Specialty deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async restoreSpecialty(req, res, next) {
    try {
      const { id } = req.params;
      const userId = req.user.id;

      await specialtiesService.restoreSpecialty(parseInt(id), userId);

      return successResponse(res, null, 'Specialty restored successfully');
    } catch (error) {
      next(error);
    }
  }

  async getDeletedSpecialties(req, res, next) {
    try {
      const { page = 1, limit = 20, name } = req.query;

      const filters = {};
      if (name) filters.name = name;

      const result = await specialtiesService.getDeletedSpecialties(filters, {
        page: parseInt(page),
        limit: parseInt(limit)
      });

      return successResponse(res, result, 'Deleted specialties retrieved successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new SpecialtiesController();