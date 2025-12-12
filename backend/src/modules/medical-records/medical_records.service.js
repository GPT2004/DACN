const prisma = require('../../config/database');
const { ROLES } = require('../../config/constants');
const notificationsService = require('../notifications/notifications.service');
const emailService = require('../../utils/email');
const patientsService = require('../patients/patients.service');

class MedicalRecordService {
  async createMedicalRecord(data, user) {
    console.log('📝 [CREATE MEDICAL RECORD] Input data:', data);
    
    // Verify doctor
    let doctor_id = data.doctor_id;
    if (user && user.role && user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor) {
        throw new Error('Doctor profile not found');
      }
      doctor_id = doctor.id;
      console.log('👨‍⚕️ Doctor ID resolved:', doctor_id);
    }

    // Verify patient exists
    const patient = await prisma.patients.findUnique({
      where: { id: parseInt(data.patient_id) },
    });

    if (!patient) {
      throw new Error('Patient not found');
    }
    console.log('👤 Patient found:', patient.id);

    // Verify appointment if provided
    if (data.appointment_id) {
      const appointment = await prisma.appointments.findUnique({
        where: { id: parseInt(data.appointment_id) },
      });

      if (!appointment) {
        throw new Error('Appointment not found');
      }

      if (appointment.patient_id !== parseInt(data.patient_id)) {
        throw new Error('Appointment does not belong to this patient');
      }
      console.log('📅 Appointment verified:', appointment.id);
    }

    // Ensure we have a valid doctor id (medical_records.doctor_id is non-nullable)
    if (!doctor_id) {
      throw new Error('Doctor id is required to create a medical record');
    }

    const resolvedDoctorId = parseInt(doctor_id);
    if (isNaN(resolvedDoctorId)) {
      throw new Error('Invalid doctor id');
    }

    // Create medical record
    const createData = {
      // connect patient relation explicitly (Prisma requires the relation)
      patient: { connect: { id: parseInt(data.patient_id) } },
      // connect doctor relation explicitly so Prisma accepts the required relation
      doctor: { connect: { id: resolvedDoctorId } },
      diagnosis: data.diagnosis,
      notes: data.notes,
      exam_results: data.exam_results || null,
      lab_tests: data.lab_tests || null,
      attachments: data.attachments || null,
    };

    // If appointment_id is provided, connect the appointment relation (not use scalar)
    if (data.appointment_id) {
      createData.appointment = { connect: { id: parseInt(data.appointment_id) } };
    }

    console.log('💾 Creating medical record with data:', createData);
    
    const record = await prisma.medical_records.create({
      data: createData,
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
            appointment_time: true,
          },
        },
        lab_orders: {
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
                  },
                },
              },
            },
          },
        },
      },
    });

    console.log('✅ Medical record created:', record.id);

    // If lab_tests provided, create lab_order for this medical record
    if (data.lab_tests && data.lab_tests.trim()) {
      try {
        console.log('🧪 Creating lab order with tests:', data.lab_tests);
        await prisma.lab_orders.create({
          data: {
            medical_record_id: record.id,
            appointment_id: data.appointment_id ? parseInt(data.appointment_id) : null,
            patient_id: parseInt(data.patient_id),
            doctor_id: resolvedDoctorId,
            tests: { description: data.lab_tests },
            status: 'PENDING',
          },
        });
        console.log('✅ Lab order created');
      } catch (err) {
        console.error('❌ Failed to create lab order:', err.message);
        // Don't fail the medical record creation if lab order creation fails
      }
    }

    return record;
  }

  async getMedicalRecords(filters, user) {
    const { patient_id, doctor_id, page = 1, limit = 10, q } = filters || {};
    const skip = (page - 1) * limit;

    const where = {};

    // text search q: prefix match diagnosis or patient full_name
    if (q && q.trim()) {
      const qTrim = q.trim();
      where.OR = [
        { diagnosis: { startsWith: qTrim, mode: 'insensitive' } },
        { patient: { user: { full_name: { startsWith: qTrim, mode: 'insensitive' } } } },
        { patient: { phone: { startsWith: qTrim, mode: 'insensitive' } } },
        { patient: { user: { phone: { startsWith: qTrim, mode: 'insensitive' } } } },
      ];
    }

    // Role-based filtering
    if (user && user.role && user.role.name === ROLES.PATIENT) {
      // Allow patient access for all patient rows linked to this user
      const patients = await prisma.patients.findMany({
        where: { OR: [{ user_id: user.id }, { owner_user_id: user.id }] },
        select: { id: true },
      });
      const patientIds = patients.map((p) => p.id);
      // If none found, set an empty IN clause to return no records
      where.patient_id = { in: patientIds };
    } else if (user && user.role && user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (doctor) {
        where.doctor_id = doctor.id;
      }
    }

    // Additional filters
    if (patient_id) where.patient_id = parseInt(patient_id);
    if (doctor_id) where.doctor_id = parseInt(doctor_id);

    const [records, total] = await Promise.all([
      prisma.medical_records.findMany({
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
              appointment_time: true,
            },
          },
          lab_orders: {
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
                    },
                  },
                },
              },
            },
          },
          prescriptions: {
            select: {
              id: true,
              status: true,
              total_amount: true,
              items: true,
              created_at: true,
            },
          },
        },
        orderBy: {
          created_at: 'desc',
        },
      }),
      prisma.medical_records.count({ where }),
    ]);

    return {
      records,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  async getMedicalRecordById(id, user) {
    const record = await prisma.medical_records.findUnique({
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
            appointment_time: true,
            reason: true,
            timeslot_id: true,
            status: true,
          },
        },
        lab_orders: {
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
                  },
                },
              },
            },
          },
        },
        prescriptions: {
          select: {
            id: true,
            status: true,
            total_amount: true,
            items: true,
            created_at: true,
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
          },
        },
      },
    });

    if (!record) {
      throw new Error('Medical record not found');
    }

    // Authorization check
    if (user && user.role && user.role.name === ROLES.PATIENT) {
      // Gather all patient rows related to this user (either direct or owned)
      const patients = await prisma.patients.findMany({
        where: { OR: [{ user_id: user.id }, { owner_user_id: user.id }] },
        select: { id: true },
      });
      const patientIds = patients.map((p) => p.id);
      if (!patientIds.includes(record.patient_id)) {
        throw new Error('Unauthorized access');
      }
    } else if (user && user.role && user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || record.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    return record;
  }

  async updateMedicalRecord(id, data, user) {
    const record = await prisma.medical_records.findUnique({
      where: { id: parseInt(id) },
    });

    if (!record) {
      throw new Error('Medical record not found');
    }

    // Authorization check - only the doctor who created it can update
    if (user && user.role && user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || record.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    const updateData = {};
    if (data.diagnosis !== undefined) updateData.diagnosis = data.diagnosis;
    if (data.notes !== undefined) updateData.notes = data.notes;
    if (data.exam_results !== undefined) updateData.exam_results = data.exam_results;
    if (data.lab_tests !== undefined) updateData.lab_tests = data.lab_tests;
    if (data.attachments !== undefined) updateData.attachments = data.attachments;
    updateData.updated_at = new Date();

    const updated = await prisma.medical_records.update({
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
        lab_orders: {
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
                  },
                },
              },
            },
          },
        },
        prescriptions: {
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
                  },
                },
              },
            },
          },
        },
      },
    });

    // If lab_tests provided, update or create lab_order for this medical record
    if (data.lab_tests !== undefined && data.lab_tests) {
      try {
        const existingLabOrder = await prisma.lab_orders.findFirst({
          where: { medical_record_id: parseInt(id) },
        });

        if (existingLabOrder) {
          await prisma.lab_orders.update({
            where: { id: existingLabOrder.id },
            data: {
              tests: { description: data.lab_tests },
              updated_at: new Date(),
            },
          });
        } else if (typeof data.lab_tests === 'string' && data.lab_tests.trim()) {
          await prisma.lab_orders.create({
            data: {
              medical_record_id: parseInt(id),
              patient_id: updated.patient_id,
              doctor_id: updated.doctor_id,
              appointment_id: updated.appointment_id,
              tests: { description: data.lab_tests },
              status: 'PENDING',
            },
          });
        }
      } catch (err) {
        console.error('Failed to update lab order:', err.message);
        // Don't fail the medical record update if lab order update fails
      }
    }

    return updated;
  }

  async deleteMedicalRecord(id, user) {
    const record = await prisma.medical_records.findUnique({
      where: { id: parseInt(id) },
    });

    if (!record) {
      throw new Error('Medical record not found');
    }

    // Allow deletion by administrators or by the doctor who created the record
    if (!user || !user.role) {
      throw new Error('Only administrators or the creating doctor can delete medical records');
    }

    if (user.role.name === ROLES.ADMIN) {
      // admins may delete
    } else if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({ where: { user_id: user.id } });
      if (!doctor || record.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    } else {
      throw new Error('Only administrators or the creating doctor can delete medical records');
    }

    await prisma.medical_records.delete({
      where: { id: parseInt(id) },
    });

    return { message: 'Medical record deleted successfully' };
  }

  async getPatientMedicalHistory(patient_id, user) {
    // Authorization check
    if (user && user.role && user.role.name === ROLES.PATIENT) {
      const patients = await prisma.patients.findMany({
        where: { OR: [{ user_id: user.id }, { owner_user_id: user.id }] },
        select: { id: true },
      });
      const patientIds = patients.map((p) => p.id);
      if (!patientIds.includes(parseInt(patient_id))) {
        throw new Error('Unauthorized access');
      }
    }

    const records = await prisma.medical_records.findMany({
      where: {
        patient_id: parseInt(patient_id),
      },
      include: {
        doctor: {
          include: {
            user: {
              select: {
                full_name: true,
                specialties: true,
              },
            },
          },
        },
        appointment: {
          select: {
            appointment_date: true,
            appointment_time: true,
            reason: true,
          },
        },
      },
      orderBy: {
        created_at: 'desc',
      },
    });

    // Get patient info
    const patient = await prisma.patients.findUnique({
      where: { id: parseInt(patient_id) },
      include: {
        user: {
          select: {
            full_name: true,
            dob: true,
            phone: true,
            email: true,
          },
        },
      },
    });

    return {
      patient,
      records,
      total_visits: records.length,
    };
  }

  async sendMedicalRecordToPatient(id, user) {
    const record = await prisma.medical_records.findUnique({
      where: { id: parseInt(id) },
      include: {
        patient: { include: { user: true } },
        doctor: { include: { user: true } },
      },
    });

    if (!record) throw new Error('Medical record not found');

    // Authorization: only admin or creating doctor may send
    if (!user || !user.role) throw new Error('Unauthorized');
    if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({ where: { user_id: user.id } });
      if (!doctor || record.doctor_id !== doctor.id) throw new Error('Unauthorized');
    } else if (user.role.name !== ROLES.ADMIN) {
      throw new Error('Only admin or creating doctor can send medical records to patients');
    }

    // Determine recipient user id (patient.user.id or owner_user_id)
    const patient = record.patient;
    const recipientUserId = (patient && patient.user && patient.user.id) ? patient.user.id : (patient && patient.owner_user_id) ? patient.owner_user_id : null;
    if (!recipientUserId) throw new Error('No linked user found for this patient');

    // Resolve or create a patient row for the recipient user so the record can be attached/shared
    const recipientPatient = await patientsService.ensurePatientForUser(recipientUserId);
    if (!recipientPatient || !recipientPatient.id) throw new Error('Failed to resolve recipient patient row');

    const payload = {
      medical_record_id: record.id,
      doctor_name: record.doctor?.user?.full_name || null,
      patient_id: record.patient_id,
      message: `Bác sĩ đã gửi hồ sơ bệnh án #${record.id}. Mở để xem chi tiết.`,
      url: `/patient/medical-records?scroll_to=${record.id}`,
    };

    // create notification record for patient (keep in-app notification)
    await notificationsService.createNotification({ user_id: recipientUserId, type: 'medical_record_shared', payload });

    // Create an entry in shared_medical_records so the patient sees this record in their UI
    try {
      await prisma.$executeRaw`
        INSERT INTO "shared_medical_records" (medical_record_id, recipient_patient_id, shared_by_user_id, created_at)
        VALUES (${record.id}, ${recipientPatient.id}, ${user ? user.id : null}, now())
        ON CONFLICT (medical_record_id, recipient_patient_id) DO NOTHING;
      `;
    } catch (e) {
      console.warn('Failed to create shared_medical_records entry:', e.message || e);
    }

    // Do NOT send email by default when sharing to patient UI (user requested UI-share only)

    return { success: true, sent_to: recipientUserId, shared_with_patient_id: recipientPatient.id };
  }

  async addAttachment(id, file, user) {
    const record = await prisma.medical_records.findUnique({
      where: { id: parseInt(id) },
    });

    if (!record) {
      throw new Error('Medical record not found');
    }

    // Authorization check
    if (user && user.role && user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (!doctor || record.doctor_id !== doctor.id) {
        throw new Error('Unauthorized access');
      }
    }

    // Get existing attachments
    let attachments = record.attachments || [];
    if (typeof attachments === 'string') {
      attachments = JSON.parse(attachments);
    }

    // Add new attachment
    const newAttachment = {
      filename: file.originalname,
      path: file.path,
      mimetype: file.mimetype,
      size: file.size,
      uploaded_at: new Date(),
    };

    attachments.push(newAttachment);

    // Update record
    const updated = await prisma.medical_records.update({
      where: { id: parseInt(id) },
      data: {
        attachments: attachments,
        updated_at: new Date(),
      },
    });

    return updated;
  }
}

module.exports = new MedicalRecordService();