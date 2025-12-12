const prisma = require('../../config/database');
const { ROLES, LAB_STATUS } = require('../../config/constants');

class LabOrderService {
  async createLabOrder(data, user) {
    // Verify doctor
    let doctor_id = data.doctor_id;
    if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor) {
        throw new Error('Doctor profile not found');
      }
      doctor_id = doctor.id;
    }

    // Verify patient
    const patient = await prisma.patients.findUnique({
      where: { id: parseInt(data.patient_id) },
    });

    if (!patient) {
      throw new Error('Patient not found');
    }

    // Verify appointment if provided
    if (data.appointment_id) {
      const appointment = await prisma.appointments.findUnique({
        where: { id: parseInt(data.appointment_id) },
      });

      if (!appointment) {
        throw new Error('Appointment not found');
      }
    }

    // Create lab order
    const labOrder = await prisma.lab_orders.create({
      data: {
        appointment_id: data.appointment_id ? parseInt(data.appointment_id) : null,
        medical_record_id: data.medical_record_id ? parseInt(data.medical_record_id) : null,
        patient_id: parseInt(data.patient_id),
        doctor_id: parseInt(doctor_id),
        tests: data.tests,
        status: LAB_STATUS.PENDING,
        results: null,
      },
      include: {
        patient: {
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                dob: true,
              },
            },
          },
        },
        doctor: {
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
              },
            },
          },
        },
        appointment: {
          select: {
            appointment_date: true,
          },
        },
      },
    });

    return labOrder;
  }

  async getLabOrders(filters, user) {
    const { patient_id, doctor_id, status, page, limit } = filters;
    const skip = (page - 1) * limit;

    const where = {};

    // Role-based filtering
    if (user.role.name === ROLES.PATIENT) {
      const patient = await prisma.patients.findFirst({
        where: { user_id: user.id },
      });
      if (patient) {
        where.patient_id = patient.id;
      }
    } else if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (doctor) {
        where.doctor_id = doctor.id;
      }
    } else if (user.role.name === ROLES.LAB_TECH) {
      // Lab tech sees all pending and in-progress orders
      where.status = {
        in: [LAB_STATUS.PENDING, LAB_STATUS.IN_PROGRESS],
      };
    }

    // Additional filters
    if (patient_id) where.patient_id = parseInt(patient_id);
    if (doctor_id) where.doctor_id = parseInt(doctor_id);
    if (status) where.status = status;

    const [labOrders, total] = await Promise.all([
      prisma.lab_orders.findMany({
        where,
        skip,
        take: limit,
        include: {
          patient: {
            include: {
              user: {
                select: {
                  id: true,
                  full_name: true,
                  dob: true,
                },
              },
            },
          },
          doctor: {
            include: {
              user: {
                select: {
                  id: true,
                  full_name: true,
                },
              },
            },
          },
          appointment: {
            select: {
              appointment_date: true,
            },
          },
        },
        orderBy: {
          created_at: 'desc',
        },
      }),
      prisma.lab_orders.count({ where }),
    ]);

    return {
      labOrders,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  async getLabOrderById(id, user) {
    const labOrder = await prisma.lab_orders.findUnique({
      where: { id: parseInt(id) },
      include: {
        patient: {
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                phone: true,
                email: true,
                dob: true,
              },
            },
          },
        },
        doctor: {
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                phone: true,
              },
            },
          },
        },
        appointment: {
          select: {
            id: true,
            appointment_date: true,
            reason: true,
          },
        },
      },
    });

    if (!labOrder) {
      throw new Error('Lab order not found');
    }

    // Authorization check
    if (user.role.name === ROLES.PATIENT) {
      const patient = await prisma.patients.findFirst({
        where: { user_id: user.id },
      });
      if (!patient || labOrder.patient_id !== patient.id) {
        throw new Error('Unauthorized access');
      }
    } else if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || labOrder.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    return labOrder;
  }

  async updateLabOrder(id, data, user) {
    const labOrder = await prisma.lab_orders.findUnique({
      where: { id: parseInt(id) },
    });

    if (!labOrder) {
      throw new Error('Lab order not found');
    }

    // Can only update PENDING orders
    if (labOrder.status !== LAB_STATUS.PENDING) {
      throw new Error('Can only update pending lab orders');
    }

    // Authorization check - must be the doctor who created it
    if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || labOrder.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    const updateData = {};
    if (data.tests) updateData.tests = data.tests;
    if (data.status) updateData.status = data.status;
    updateData.updated_at = new Date();

    const updated = await prisma.lab_orders.update({
      where: { id: parseInt(id) },
      data: updateData,
      include: {
        patient: {
          include: {
            user: {
              select: {
                full_name: true,
              },
            },
          },
        },
        doctor: {
          include: {
            user: {
              select: {
                full_name: true,
              },
            },
          },
        },
      },
    });

    return updated;
  }

  async updateLabResults(id, data) {
    const labOrder = await prisma.lab_orders.findUnique({
      where: { id: parseInt(id) },
    });

    if (!labOrder) {
      throw new Error('Lab order not found');
    }

    // Update results
    const updateData = {
      results: data.results,
      status: LAB_STATUS.IN_PROGRESS,
      updated_at: new Date(),
    };

    const updated = await prisma.lab_orders.update({
      where: { id: parseInt(id) },
      data: updateData,
      include: {
        patient: {
          include: {
            user: true,
          },
        },
        doctor: {
          include: {
            user: true,
          },
        },
      },
    });

    return updated;
  }

  async completeLabOrder(id) {
    const labOrder = await prisma.lab_orders.findUnique({
      where: { id: parseInt(id) },
    });

    if (!labOrder) {
      throw new Error('Lab order not found');
    }

    if (labOrder.status === LAB_STATUS.COMPLETED) {
      throw new Error('Lab order already completed');
    }

    if (!labOrder.results) {
      throw new Error('Cannot complete lab order without results');
    }

    const updated = await prisma.lab_orders.update({
      where: { id: parseInt(id) },
      data: {
        status: LAB_STATUS.COMPLETED,
        updated_at: new Date(),
      },
      include: {
        patient: {
          include: {
            user: true,
          },
        },
        doctor: {
          include: {
            user: true,
          },
        },
      },
    });

    return updated;
  }

  async deleteLabOrder(id, user) {
    const labOrder = await prisma.lab_orders.findUnique({
      where: { id: parseInt(id) },
    });

    if (!labOrder) {
      throw new Error('Lab order not found');
    }

    // Can only delete PENDING orders
    if (labOrder.status !== LAB_STATUS.PENDING) {
      throw new Error('Can only delete pending lab orders');
    }

    // Authorization check
    if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || labOrder.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    await prisma.lab_orders.delete({
      where: { id: parseInt(id) },
    });

    return { message: 'Lab order deleted successfully' };
  }
}

module.exports = new LabOrderService();