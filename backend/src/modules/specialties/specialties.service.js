/* eslint-disable no-console */
const prisma = require('../../config/database');

class SpecialtiesService {
  async getAllSpecialties(filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {
        deleted_at: null
      };

      if (filters.name) {
        where.name = {
          contains: filters.name,
          mode: 'insensitive'
        };
      }

      const [total, specialties] = await Promise.all([
        prisma.specialties.count({ where }),
        prisma.specialties.findMany({
          where,
          skip,
          take: limit,
          include: {
            _count: {
              select: {
                doctorSpecialties: true
              }
            }
          },
          orderBy: {
            name: 'asc'
          }
        })
      ]);

      return {
        specialties,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getAllSpecialties:', error);
      throw error;
    }
  }

  async getSpecialtyById(id) {
    try {
      const specialty = await prisma.specialties.findUnique({
        where: { id, deleted_at: null },
        include: {
          _count: {
            select: {
              doctorSpecialties: true
            }
          }
        }
      });

      return specialty;
    } catch (error) {
      console.error('Error in getSpecialtyById:', error);
      throw error;
    }
  }

  async createSpecialty(specialtyData) {
    try {
      const { name, slug, description, image_url } = specialtyData;

      // Check if specialty name already exists
      const existingSpecialty = await prisma.specialties.findFirst({
        where: {
          name: {
            equals: name,
            mode: 'insensitive'
          },
          deleted_at: null
        }
      });

      if (existingSpecialty) {
        throw new Error('Specialty name already exists');
      }

      const specialty = await prisma.specialties.create({
        data: {
          name,
          slug,
          description,
          image_url
        }
      });

      return specialty;
    } catch (error) {
      console.error('Error in createSpecialty:', error);
      throw error;
    }
  }

  async updateSpecialty(id, updateData) {
    try {
      const specialty = await prisma.specialties.findUnique({
        where: { id, deleted_at: null }
      });

      if (!specialty) {
        return null;
      }

      // Check if new name conflicts with existing specialty
      if (updateData.name && updateData.name !== specialty.name) {
        const existingSpecialty = await prisma.specialties.findFirst({
          where: {
            name: {
              equals: updateData.name,
              mode: 'insensitive'
            },
            id: {
              not: id
            },
            deleted_at: null
          }
        });

        if (existingSpecialty) {
          throw new Error('Specialty name already exists');
        }
      }

      const updatedSpecialty = await prisma.specialties.update({
        where: { id },
        data: updateData
      });

      return updatedSpecialty;
    } catch (error) {
      console.error('Error in updateSpecialty:', error);
      throw error;
    }
  }

  async deleteSpecialty(id, userId) {
    try {
      const specialty = await prisma.specialties.findUnique({
        where: { id, deleted_at: null },
        include: {
          _count: {
            select: {
              doctorSpecialties: true
            }
          }
        }
      });

      if (!specialty) {
        throw new Error('Specialty not found');
      }

      // Check if specialty has doctors assigned
      if (specialty._count.doctorSpecialties > 0) {
        throw new Error('Cannot delete specialty with assigned doctors');
      }

      // Soft delete
      await prisma.specialties.update({
        where: { id },
        data: {
          deleted_at: new Date()
        }
      });

      // Log audit
      await prisma.audit_logs.create({
        data: {
          user_id: userId,
          action: 'DELETE_SPECIALTY',
          meta: {
            specialty_id: id,
            specialty_name: specialty.name
          }
        }
      });
    } catch (error) {
      console.error('Error in deleteSpecialty:', error);
      throw error;
    }
  }

  async restoreSpecialty(id, userId) {
    try {
      const specialty = await prisma.specialties.findUnique({
        where: { id }
      });

      if (!specialty) {
        throw new Error('Specialty not found');
      }

      if (!specialty.deleted_at) {
        throw new Error('Specialty is not deleted');
      }

      await prisma.specialties.update({
        where: { id },
        data: {
          deleted_at: null
        }
      });

      // Log audit
      await prisma.audit_logs.create({
        data: {
          user_id: userId,
          action: 'RESTORE_SPECIALTY',
          meta: {
            specialty_id: id,
            specialty_name: specialty.name
          }
        }
      });
    } catch (error) {
      console.error('Error in restoreSpecialty:', error);
      throw error;
    }
  }

  async getDeletedSpecialties(filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {
        deleted_at: {
          not: null
        }
      };

      if (filters.name) {
        where.name = {
          contains: filters.name,
          mode: 'insensitive'
        };
      }

      const [total, specialties] = await Promise.all([
        prisma.specialties.count({ where }),
        prisma.specialties.findMany({
          where,
          skip,
          take: limit,
          orderBy: {
            deleted_at: 'desc'
          }
        })
      ]);

      return {
        specialties,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getDeletedSpecialties:', error);
      throw error;
    }
  }
}

module.exports = new SpecialtiesService();