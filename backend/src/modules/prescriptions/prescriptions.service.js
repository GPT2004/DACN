const prisma = require('../../config/database');
const { ROLES, PRESCRIPTION_STATUS } = require('../../config/constants');
const notificationsService = require('../notifications/notifications.service');

class PrescriptionService {
  async createPrescription(data, user) {
    // Verify doctor
    let doctor_id = data.doctor_id;
    if (!doctor_id) {
      // If no doctor_id provided, try to get it from user
      if (user.role.name === ROLES.DOCTOR) {
        const doctor = await prisma.doctors.findFirst({
          where: { user_id: user.id },
        });
        if (!doctor) {
          throw new Error('Doctor profile not found');
        }
        doctor_id = doctor.id;
      } else {
        throw new Error('doctor_id is required');
      }
    } else {
      // Validate that the doctor_id matches the current user if user is a doctor
      if (user.role.name === ROLES.DOCTOR) {
        const doctor = await prisma.doctors.findFirst({
          where: { user_id: user.id },
        });
        if (!doctor || doctor.id !== parseInt(doctor_id)) {
          throw new Error('Unauthorized: cannot create prescription for a different doctor');
        }
      }
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

    // Verify medicines and calculate total
    let total_amount = 0;
    const items = [];

    for (const item of data.items) {
      // Try to resolve medicine by id first, then by name. If not found, fall back to using provided name/unit_price.
      let medicine = null;
      const medId = item.medicine_id ? parseInt(item.medicine_id) : null;
      try {
        if (medId) {
          medicine = await prisma.medicines.findUnique({ where: { id: medId } });
        }
        if (!medicine && item.medicine_name) {
          medicine = await prisma.medicines.findFirst({ where: { name: item.medicine_name } });
        }
      } catch (err) {
        // ignore and handle below
        medicine = null;
      }

      const resolvedName = medicine ? medicine.name : (item.medicine_name || 'Unknown');
      const unit_price = item.unit_price || (medicine ? medicine.price : 0);
      const qty = Number(item.quantity) || 0;
      const itemTotal = unit_price * qty;

      items.push({
        medicine_id: medicine ? medicine.id : (medId || null),
        medicine_name: resolvedName,
        quantity: qty,
        unit_price: unit_price,
        instructions: item.instructions || '',
        dosage: item.dosage || '',
      });

      total_amount += itemTotal;
    }

    // Create prescription
    const createData = {
      appointment_id: data.appointment_id ? parseInt(data.appointment_id) : null,
      medical_record_id: data.medical_record_id ? parseInt(data.medical_record_id) : null,
      doctor_id: parseInt(doctor_id),
      patient_id: parseInt(data.patient_id),
      items: items,
      total_amount: total_amount,
      status: PRESCRIPTION_STATUS.DRAFT,
    };

    const prescription = await prisma.prescriptions.create({
      data: createData,
      include: {
        doctor: {
          include: {
            user: {
              select: {
                full_name: true,
              },
            },
          },
        },
        patient: {
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

    return prescription;
  }

  async getPrescriptions(filters, user) {
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
    } else if (user.role.name === ROLES.PHARMACIST) {
      // Pharmacist sees prescriptions ready for dispensing:
      // - APPROVED: approved by doctor but not yet invoiced (manual workflow)
      // - INVOICED: invoice was paid, ready for dispensing (main workflow)
      // NOTE: temporary hotfix uses INVOICED instead of READY_FOR_DISPENSE
      // to avoid DB enum mismatch. Apply migration to add READY_FOR_DISPENSE
      // and restore the intended enum value.
      where.status = { in: [PRESCRIPTION_STATUS.APPROVED, PRESCRIPTION_STATUS.INVOICED] };
    }

    // Additional filters
    if (patient_id) where.patient_id = parseInt(patient_id);
    if (doctor_id) where.doctor_id = parseInt(doctor_id);
    if (status) where.status = status;

    const [prescriptions, total] = await Promise.all([
      prisma.prescriptions.findMany({
        where,
        skip,
        take: limit,
        include: {
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
          patient: {
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
              appointment_date: true,
            },
          },
        },
        orderBy: {
          created_at: 'desc',
        },
      }),
      prisma.prescriptions.count({ where }),
    ]);

    return {
      prescriptions,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  async getPrescriptionById(id, user) {
    const prescription = await prisma.prescriptions.findUnique({
      where: { id: parseInt(id) },
      include: {
        doctor: {
          select: {
            consultation_fee: true,
            user: {
              select: {
                id: true,
                full_name: true,
                phone: true,
              },
            },
          },
        },
        patient: {
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                phone: true,
                dob: true,
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

    if (!prescription) {
      throw new Error('Prescription not found');
    }

    // Authorization check
    if (user.role.name === ROLES.PATIENT) {
      const patient = await prisma.patients.findFirst({
        where: { user_id: user.id },
      });
      if (!patient || prescription.patient_id !== patient.id) {
        throw new Error('Unauthorized access');
      }
    } else if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || prescription.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    return prescription;
  }

  async updatePrescription(id, data, user) {
    const prescription = await prisma.prescriptions.findUnique({
      where: { id: parseInt(id) },
    });

    if (!prescription) {
      throw new Error('Prescription not found');
    }

    // Can only update DRAFT prescriptions
    if (prescription.status !== PRESCRIPTION_STATUS.DRAFT) {
      throw new Error('Can only update draft prescriptions');
    }

    // Authorization check
    if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || prescription.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    const updateData = {};

    // Update items if provided
    if (data.items) {
      let total_amount = 0;
      const items = [];

      for (const item of data.items) {
        let medicine = null;
        const medId = item.medicine_id ? parseInt(item.medicine_id) : null;
        try {
          if (medId) {
            medicine = await prisma.medicines.findUnique({ where: { id: medId } });
          }
          if (!medicine && item.medicine_name) {
            medicine = await prisma.medicines.findFirst({ where: { name: item.medicine_name } });
          }
        } catch (err) {
          medicine = null;
        }

        const resolvedName = medicine ? medicine.name : (item.medicine_name || 'Unknown');
        const unit_price = item.unit_price || (medicine ? medicine.price : 0);
        const qty = Number(item.quantity) || 0;
        const itemTotal = unit_price * qty;

        items.push({
          medicine_id: medicine ? medicine.id : (medId || null),
          medicine_name: resolvedName,
          quantity: qty,
          unit_price: unit_price,
          instructions: item.instructions || '',
          dosage: item.dosage || '',
        });

        total_amount += itemTotal;
      }

      updateData.items = items;
      updateData.total_amount = total_amount;
    }

    updateData.updated_at = new Date();

    const updated = await prisma.prescriptions.update({
      where: { id: parseInt(id) },
      data: updateData,
      include: {
        doctor: {
          include: {
            user: {
              select: {
                full_name: true,
              },
            },
          },
        },
        patient: {
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

  async approvePrescription(id, user) {
    const prescription = await prisma.prescriptions.findUnique({
      where: { id: parseInt(id) },
    });

    if (!prescription) {
      throw new Error('Prescription not found');
    }

    if (prescription.status !== PRESCRIPTION_STATUS.DRAFT) {
      throw new Error('Only draft prescriptions can be approved');
    }

    // Authorization check - must be the doctor who created it
    if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || prescription.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    const updated = await prisma.prescriptions.update({
      where: { id: parseInt(id) },
      data: {
        status: PRESCRIPTION_STATUS.APPROVED,
        updated_at: new Date(),
      },
      include: {
        patient: {
          include: {
            user: true,
          },
        },
      },
    });

    return updated;
  }

  async dispensePrescription(id, user = null, payload = {}) {
    const prescription = await prisma.prescriptions.findUnique({
      where: { id: parseInt(id) },
    });

    if (!prescription) {
      throw new Error('Prescription not found');
    }

    // Allow dispensing when prescription is DRAFT, APPROVED or INVOICED.
    // DRAFT: pharmacist can dispense directly without approval (for testing/urgent cases)
    // APPROVED: doctor approved, ready for invoicing
    // INVOICED: receipt paid invoice, prescription is INVOICED
    const allowedStatuses = [PRESCRIPTION_STATUS.DRAFT, PRESCRIPTION_STATUS.APPROVED, PRESCRIPTION_STATUS.INVOICED];
    if (!allowedStatuses.includes(prescription.status)) {
      throw new Error('Only draft, approved or invoiced prescriptions can be dispensed');
    }

    // Determine dispense items: either provided by pharmacist (payload.items)
    // or fall back to prescription.items recorded earlier.
    const prescriptionItems = typeof prescription.items === 'string'
      ? JSON.parse(prescription.items)
      : prescription.items;

    // Map of medicine_id -> quantity to dispense
    const dispenseMap = new Map();
    if (payload && Array.isArray(payload.items) && payload.items.length > 0) {
      for (const it of payload.items) {
        const medId = parseInt(it.medicine_id);
        const qty = Number(it.dispense_quantity) || 0;
        if (medId && qty > 0) dispenseMap.set(medId, qty);
      }
    } else {
      for (const it of prescriptionItems) {
        const medId = it.medicine_id ? parseInt(it.medicine_id) : null;
        const qty = Number(it.quantity) || 0;
        if (medId && qty > 0) dispenseMap.set(medId, qty);
      }
    }

    // Now iterate over prescriptionItems to preserve order and names for errors
    for (const item of prescriptionItems) {
      const medId = item.medicine_id ? parseInt(item.medicine_id) : null;
      const medName = item.medicine_name || 'Unknown';
      const requiredQty = dispenseMap.has(medId) ? dispenseMap.get(medId) : 0;
      if (!medId || requiredQty <= 0) continue; // nothing to dispense for this line

      // Get available stock
      const stocks = await prisma.stocks.findMany({
        where: {
          medicine_id: medId,
          expiry_date: { gt: new Date() },
          quantity: { gt: 0 },
        },
        orderBy: { expiry_date: 'asc' },
      });

      let remainingQty = requiredQty;
      for (const stock of stocks) {
        if (remainingQty <= 0) break;
        const deductQty = Math.min(stock.quantity, remainingQty);
        await prisma.stocks.update({
          where: { id: stock.id },
          data: { quantity: stock.quantity - deductQty, updated_at: new Date() },
        });
        remainingQty -= deductQty;
      }

      if (remainingQty > 0) {
        throw new Error(`Insufficient stock for medicine: ${medName}`);
      }
    }

    // Update prescription status
    const updated = await prisma.prescriptions.update({
      where: { id: parseInt(id) },
      data: {
        status: PRESCRIPTION_STATUS.DISPENSED,
        updated_at: new Date(),
      },
    });

    return updated;
  }

  async deletePrescription(id, user) {
    const prescription = await prisma.prescriptions.findUnique({
      where: { id: parseInt(id) },
    });

    if (!prescription) {
      throw new Error('Prescription not found');
    }

    // Can only delete DRAFT prescriptions
    if (prescription.status !== PRESCRIPTION_STATUS.DRAFT) {
      throw new Error('Can only delete draft prescriptions');
    }

    // Authorization check
    if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || prescription.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    await prisma.prescriptions.delete({
      where: { id: parseInt(id) },
    });

    return { message: 'Prescription deleted successfully' };
  }

  async notifyReception(id, user) {
    const prescription = await prisma.prescriptions.findUnique({ where: { id: parseInt(id) }, include: { doctor: { include: { user: true } }, patient: { include: { user: true } } } });
    if (!prescription) throw new Error('Prescription not found');

    // Only doctor or admin can notify
    if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({ where: { user_id: user.id } });
      if (!doctor || prescription.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    // Update prescription with notified info
    let updated;
    try {
      updated = await prisma.prescriptions.update({
        where: { id: parseInt(id) },
        data: {
          notified_at: new Date(),
          notified_by: user.id,
          updated_at: new Date(),
        },
        include: {
          doctor: { include: { user: true } },
          patient: { include: { user: true } },
        },
      });
    } catch (err) {
      // If DB schema/migration hasn't been applied (or Prisma client mismatch), fall back to
      // updating only the updated_at field so the operation can continue and notification is sent.
      console.error('prescriptions.update failed, falling back to minimal update:', err.message || err);
      try {
        await prisma.prescriptions.update({ where: { id: parseInt(id) }, data: { updated_at: new Date() } });
        // refetch prescription to build payload
        updated = await prisma.prescriptions.findUnique({ where: { id: parseInt(id) }, include: { doctor: { include: { user: true } }, patient: { include: { user: true } } } });
      } catch (err2) {
        console.error('Fallback update also failed:', err2.message || err2);
        throw err2;
      }
    }

    // Broadcast notification to receptionists
    const payload = {
      prescription_id: updated.id,
      doctor_name: updated.doctor?.user?.full_name || null,
      patient_name: updated.patient?.user?.full_name || null,
      message: `Đơn thuốc #${updated.id} đã được gửi từ bác sĩ. Vui lòng tạo hóa đơn.`,
      url: `/receptionist/prescriptions-inbox?prescription_id=${updated.id}`,
    };

    await notificationsService.broadcastNotification({ role_name: 'Receptionist', type: 'prescription_for_invoice', payload });

    return updated;
  }

  async getForInvoicing(filters = { page: 1, limit: 20 }, user) {
    const { page = 1, limit = 20 } = filters;
    const skip = (page - 1) * limit;

    // Receptionist sees prescriptions that were notified
    // Only include prescriptions that have been notified and are ready for invoicing
    // (DRAFT or APPROVED status, not yet INVOICED)
    const where = {
      notified_at: { not: null },
      status: { in: [PRESCRIPTION_STATUS.DRAFT, PRESCRIPTION_STATUS.APPROVED] },
    };
    try {
      const [prescriptions, total] = await Promise.all([
        prisma.prescriptions.findMany({
          where,
          skip,
          take: limit,
          include: {
            doctor: { include: { user: true } },
            patient: { include: { user: { select: { id: true, full_name: true, phone: true } } } },
          },
          orderBy: { created_at: 'desc' },
        }),
        prisma.prescriptions.count({ where }),
      ]);

      return { prescriptions, pagination: { page, limit, total, totalPages: Math.ceil(total / limit) } };
    } catch (err) {
      console.error('Error fetching prescriptions for invoicing:', err.message || err);
      // Fail gracefully: return empty list so frontend shows "no pending prescriptions"
      return { prescriptions: [], pagination: { page, limit, total: 0, totalPages: 0 } };
    }
  }
}

module.exports = new PrescriptionService();