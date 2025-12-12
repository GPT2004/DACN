/* eslint-disable no-useless-catch */
const prisma = require('../../config/database');
const notificationsService = require('../notifications/notifications.service');
const crypto = require('crypto');
const { FRONTEND_URL, API_BASE_URL } = process.env;
const { APPOINTMENT_STATUS, ROLES } = require('../../config/constants');
const emailService = require('../../utils/email');

class AppointmentService {
  // Helper to format DATE fields to YYYY-MM-DD local date
  // NOTE: appointment_date is stored as DATE in PostgreSQL (no time component)
  // When Prisma parses it, it becomes a Date object at 00:00:00 UTC
  formatDateLocal(dateObj) {
    if (!dateObj) return null;
    
    // If it's a string (from API/external), return as-is if already formatted
    if (typeof dateObj === 'string') {
      // If it looks like a date string, return it
      if (/^\d{4}-\d{2}-\d{2}/.test(dateObj)) {
        return dateObj.split('T')[0]; // Extract YYYY-MM-DD part
      }
      return dateObj;
    }
    
    // If it's a Date object from Prisma
    if (dateObj instanceof Date) {
      // Use UTC components since DATE fields have no timezone
      const y = dateObj.getUTCFullYear();
      const m = String(dateObj.getUTCMonth() + 1).padStart(2, '0');
      const d = String(dateObj.getUTCDate()).padStart(2, '0');
      return `${y}-${m}-${d}`;
    }
    
    return null;
  }

  // Helper to format TIME fields to HH:MM local time
  // NOTE: appointment_time is stored as TIME(6) in PostgreSQL, which is local time without timezone
  // When Prisma parses it, it becomes a Date object. We need to extract the time component correctly.
  formatTimeLocal(timeObj) {
    if (!timeObj) return null;
    
    // If it's a string (from API/external), parse it directly
    if (typeof timeObj === 'string') {
      // Extract HH:MM from "HH:MM:SS" or "HH:MM"
      const match = timeObj.match(/^(\d{1,2}):(\d{1,2})/);
      if (match) {
        const h = String(match[1]).padStart(2, '0');
        const m = String(match[2]).padStart(2, '0');
        return `${h}:${m}`;
      }
      return timeObj; // Return as-is if pattern doesn't match
    }
    
    // If it's a Date object from Prisma
    if (timeObj instanceof Date) {
      // Extract time from ISO string to get the local time component
      // ISO format: 1970-01-01T08:00:00.000Z (regardless of input, JS stores as UTC)
      // The HH:MM part IS the actual time value (because PostgreSQL TIME has no timezone)
      const isoStr = timeObj.toISOString();
      const timePart = isoStr.split('T')[1]; // Get "08:00:00.000Z"
      const h = timePart.substring(0, 2);
      const m = timePart.substring(3, 5);
      return `${h}:${m}`;
    }
    
    return null;
  }

  // Helper to format appointment response
  formatAppointment(apt) {
    if (!apt) return null;
    return {
      ...apt,
      appointment_date: this.formatDateLocal(apt.appointment_date),
      appointment_time: this.formatTimeLocal(apt.appointment_time)
    };
  }

  // Helper to format appointment array response
  formatAppointments(apts) {
    if (!Array.isArray(apts)) return apts;
    return apts.map(apt => this.formatAppointment(apt));
  }

  async createAppointment(data, user) {
    try {
      // Determine patient_id based on user role and payload
      let patient_id = data.patient_id ? parseInt(data.patient_id) : undefined;
      if (user.role.name === ROLES.PATIENT) {
        if (patient_id) {
          const patient = await prisma.patients.findUnique({ where: { id: patient_id } });
          if (!patient) throw new Error('Patient not found');
          // Allow if patient belongs to this user (either linked user account or owned)
          if (!(patient.user_id === user.id || patient.owner_user_id === user.id)) {
            throw new Error('Unauthorized to create appointment for this patient');
          }
        } else {
          const patient = await prisma.patients.findFirst({ where: { user_id: user.id } });
          if (!patient) {
            console.error(`No patient profile found for user_id: ${user.id}`);
            throw new Error('Patient profile not found. Please contact support.');
          }
          patient_id = patient.id;
        }
      }

      // Use transaction to prevent double-booking
      const appointment = await prisma.$transaction(async (tx) => {
        // Lock and verify timeslot availability
        const timeslot = await tx.timeslots.findUnique({
          where: { id: parseInt(data.timeslot_id) },
        });

        if (!timeslot || !timeslot.is_active) {
          throw new Error('Timeslot not available');
        }

        if (timeslot.booked_count >= timeslot.max_patients) {
          throw new Error('Timeslot is fully booked');
        }

        // Check for duplicate appointment (same patient, same timeslot)
        const existingAppointment = await tx.appointments.findFirst({
          where: {
            patient_id: parseInt(patient_id),
            timeslot_id: parseInt(data.timeslot_id),
            status: { notIn: ['CANCELLED', 'NO_SHOW'] }
          }
        });

        if (existingAppointment) {
          throw new Error('Patient already has an appointment in this timeslot');
        }

        // Get doctor_id from timeslot
        const doctor_id = timeslot.doctor_id;

        // Determine appointment status based on user role
        let status = APPOINTMENT_STATUS.PENDING;
        let source = 'web';
        
        if (user.role.name === ROLES.RECEPTIONIST) {
          status = APPOINTMENT_STATUS.CONFIRMED;
          source = 'walk-in';
        } else if (user.role.name === ROLES.PATIENT) {
          source = 'online';
        }

        // Increment timeslot booked_count atomically only if there is capacity
        const updatedTimeslot = await tx.timeslots.updateMany({
          where: {
            id: parseInt(data.timeslot_id),
            booked_count: {
              lt: timeslot.max_patients || 1
            }
          },
          data: {
            booked_count: {
              increment: 1
            }
          }
        });

        if (!updatedTimeslot || updatedTimeslot.count === 0) {
          throw new Error('Timeslot is full');
        }

        // Create appointment
        // generate confirmation token for online bookings (patients)
        let confirmationToken = null;
        if (user.role.name === ROLES.PATIENT && status === APPOINTMENT_STATUS.PENDING) {
          confirmationToken = crypto.randomBytes(20).toString('hex');
        }

        const newAppointment = await tx.appointments.create({
          data: {
            patient_id: parseInt(patient_id),
            doctor_id: parseInt(doctor_id),
            timeslot_id: parseInt(data.timeslot_id),
            appointment_date: new Date(data.appointment_date || timeslot.date),
            appointment_time: timeslot.start_time,
            confirmation_token: confirmationToken,
            confirmation_sent_at: confirmationToken ? new Date() : null,
            status,
            reason: data.reason,
            source,
            created_by: user.id,
          },
          include: {
            patient: {
              include: {
                user: {
                  select: {
                    full_name: true,
                    phone: true,
                    email: true,
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
            timeslot: true,
          },
        });

        return newAppointment;
      });

      // After transaction: send confirmation email if needed (non-blocking)
      (async () => {
        try {
          if (appointment.confirmation_token) {
            const patientEmail = appointment.patient?.user?.email || appointment.patient?.email;
            if (patientEmail) {
              const confirmUrl = `${API_BASE_URL || FRONTEND_URL}/api/appointments/${appointment.id}/confirm?token=${appointment.confirmation_token}`;
              // pass confirmUrl via appointment object for templating
              const appointmentForEmail = { ...appointment, confirm_url: confirmUrl };
              await emailService.sendAppointmentConfirmation(appointmentForEmail, patientEmail);
            }
          } else {
            // if receptionist created and status is CONFIRMED, notify doctor/reception as needed
          }
        } catch (e) {
          try { console.error('Failed to send post-create appointment email:', e); } catch (e) {}
        }
      })();

      return this.formatAppointment(appointment);
    } catch (error) {
      throw error;
    }
  }

  async getAppointments(filters, user) {
    const { status, date, doctor_id, patient_id, page, limit } = filters;
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
    }

    // Additional filters
    if (status) where.status = status;
    if (doctor_id) where.doctor_id = parseInt(doctor_id);
    if (patient_id) where.patient_id = parseInt(patient_id);
    
    if (date) {
      // Filter by entire day instead of exact time
      const startOfDay = new Date(date);
      const endOfDay = new Date(date);
      endOfDay.setHours(23, 59, 59, 999);
      where.appointment_date = {
        gte: startOfDay,
        lte: endOfDay
      };
    }

    const [appointments, total] = await Promise.all([
      prisma.appointments.findMany({
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
                  phone: true,
                  email: true,
                },
              },
            },
          },
          // Use include for doctor relations to avoid nested select/select errors
          doctor: {
            include: {
              user: {
                select: {
                  id: true,
                  full_name: true,
                  phone: true,
                  email: true,
                },
              },
            },
          },
          timeslot: true,
        },
        orderBy: [
          { appointment_date: 'desc' },
          { appointment_time: 'desc' },
        ],
      }),
      prisma.appointments.count({ where }),
    ]);

    // Normalize each appointment to include a combined ISO datetime field
    const normalized = appointments.map((apt) => {
      try {
        const dateObj = apt.appointment_date instanceof Date ? apt.appointment_date : new Date(apt.appointment_date);

        // Determine time source: appointment_time or timeslot.start_time
        const timeSource = apt.appointment_time || (apt.timeslot && apt.timeslot.start_time) || null;

        let hours = 0, minutes = 0, seconds = 0;
        if (timeSource) {
          // If timeSource is a string like "08:00" or "08:00:00", parse it directly
          if (typeof timeSource === 'string' && /^\d{2}:\d{2}/.test(timeSource)) {
            const parts = timeSource.split(':');
            hours = parseInt(parts[0], 10);
            minutes = parseInt(parts[1], 10);
            seconds = parts[2] ? parseInt(parts[2], 10) : 0;
          } else {
            // If it's a Date object, try to extract time (but this may be wrong if stored as local time)
            const t = timeSource instanceof Date ? timeSource : new Date(timeSource);
            if (!isNaN(t.getTime())) {
              // WARNING: if appointment_time was stored as local time in DB, this will be wrong
              // For now, assume time field is stored as local time component and parse accordingly
              const timeStr = t.toISOString().split('T')[1]; // get time portion from ISO
              const timeParts = timeStr.split(':');
              hours = parseInt(timeParts[0], 10);
              minutes = parseInt(timeParts[1], 10);
              seconds = Math.floor(parseFloat(timeParts[2]));
            }
          }
        }

        // Build UTC datetime from date + time components
        const y = dateObj.getUTCFullYear();
        const mo = dateObj.getUTCMonth();
        const d = dateObj.getUTCDate();
        const dt = new Date(Date.UTC(y, mo, d, hours, minutes, seconds));

        return {
          ...apt,
          appointment_datetime: dt.toISOString(),
        };
      } catch (e) {
        console.error('[getAppointments] Error normalizing appointment:', e);
        return { ...apt };
      }
    });

    return {
      appointments: this.formatAppointments(normalized),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  async getAppointmentById(id, user) {
    // Single optimized query with all nested relations
    const appointment = await prisma.appointments.findUnique({
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
            medical_records: {
              orderBy: { created_at: 'desc' },
              select: {
                id: true,
                diagnosis: true,
                notes: true,
                created_at: true,
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
            doctorSpecialties: {
              include: {
                specialty: {
                  select: {
                    id: true,
                    name: true,
                  },
                },
              },
            },
          },
        },
        timeslot: true,
      },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    // Authorization check using fetched data
    if (user.role.name === ROLES.PATIENT) {
      if (appointment.patient?.user_id !== user.id) {
        throw new Error('Unauthorized access');
      }
    } else if (user.role.name === ROLES.DOCTOR) {
      if (appointment.doctor?.user_id !== user.id) {
        throw new Error('Unauthorized access');
      }
    }

    return this.formatAppointment(appointment);
  }

  async updateAppointment(id, data, user) {
    const appointment = await prisma.appointments.findUnique({
      where: { id: parseInt(id) },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    // Cannot update completed or cancelled appointments
    if ([APPOINTMENT_STATUS.COMPLETED, APPOINTMENT_STATUS.CANCELLED, APPOINTMENT_STATUS.NO_SHOW].includes(appointment.status)) {
      throw new Error('Cannot update completed, cancelled or no-show appointment');
    }

    const updateData = {};
    
    if (data.appointment_date) {
      updateData.appointment_date = new Date(data.appointment_date);
    }
    if (data.appointment_time) {
      updateData.appointment_time = data.appointment_time;
    }
    if (data.reason) {
      updateData.reason = data.reason;
    }
    if (data.status && [ROLES.ADMIN, ROLES.RECEPTIONIST, ROLES.DOCTOR].includes(user.role.name)) {
      updateData.status = data.status;
    }

    updateData.updated_at = new Date();

    const updated = await prisma.appointments.update({
      where: { id: parseInt(id) },
      data: updateData,
      include: {
          patient: {
            include: {
              user: {
                select: {
                  full_name: true,
                  phone: true,
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
          timeslot: true,
        },
    });

    return this.formatAppointment(updated);
  }

  async confirmAppointment(id) {
    const appointment = await prisma.appointments.findUnique({
      where: { id: parseInt(id) },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    if (appointment.status !== APPOINTMENT_STATUS.PENDING) {
      throw new Error('Only pending appointments can be confirmed');
    }

    const updated = await prisma.appointments.update({
      where: { id: parseInt(id) },
      data: {
        status: APPOINTMENT_STATUS.CONFIRMED,
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

    return this.formatAppointment(updated);
  }

  async cancelAppointment(id, reason, user) {
    const appointment = await prisma.appointments.findUnique({
      where: { id: parseInt(id) },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    if (appointment.status === APPOINTMENT_STATUS.COMPLETED) {
      throw new Error('Cannot cancel completed appointment');
    }

    // Check cancellation time (24 hours before for patients)
    if (user.role.name === ROLES.PATIENT) {
      const appointmentDateTime = new Date(appointment.appointment_date);
      const now = new Date();
      const hoursBeforeAppointment = (appointmentDateTime - now) / (1000 * 60 * 60);
      
      if (hoursBeforeAppointment < 24) {
        throw new Error('Cannot cancel appointment less than 24 hours before scheduled time. Please contact reception.');
      }
    }

    // Update appointment status (trigger will auto-decrement booked_count)
    const updated = await prisma.appointments.update({
      where: { id: parseInt(id) },
      data: {
        status: APPOINTMENT_STATUS.CANCELLED,
        reason: reason ? `${appointment.reason || ''}\nCancellation reason: ${reason}` : appointment.reason,
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

    return this.formatAppointment(updated);
  }

  // Patient confirms they will attend: set CONFIRMED and notify reception
  async patientConfirmAppointment(id, user) {
    try {
      const appointment = await prisma.appointments.findUnique({ where: { id: parseInt(id) }, include: { patient: { include: { user: true } }, doctor: { include: { user: true } } } });
      if (!appointment) throw new Error('Appointment not found');

      // Only patient owner can confirm
      if (user.role.name !== ROLES.PATIENT) {
        throw new Error('Only patients can confirm their appointments');
      }

      // Verify the appointment belongs to one of this user's patient profiles
      const patientProfile = await prisma.patients.findFirst({ where: { OR: [{ user_id: user.id }, { owner_user_id: user.id }], id: appointment.patient_id } });
      if (!patientProfile) throw new Error('Unauthorized to confirm this appointment');

      // Update status to CONFIRMED if not already
      const updated = await prisma.appointments.update({ where: { id: parseInt(id) }, data: { status: 'CONFIRMED', patient_confirmed: true, patient_confirmed_at: new Date(), confirmation_token: null, updated_at: new Date() }, include: { patient: { include: { user: true } }, doctor: { include: { user: true } } } });

      // Send confirmation email to patient (non-blocking)
      (async () => {
        try {
          const patientEmail = updated.patient?.user?.email || updated.patient?.email;
          if (patientEmail) {
            await emailService.sendAppointmentConfirmation(updated, patientEmail);
          }
        } catch (e) {
          console.error('Failed to send confirmation email to patient:', e);
        }
      })();

      // Notify receptionists via broadcast
      try {
        await notificationsService.broadcastNotification({ role_name: 'Receptionist', type: 'patient_confirmed', payload: { appointment_id: updated.id, patient_name: updated.patient?.full_name || updated.patient?.user?.full_name, doctor_name: updated.doctor?.user?.full_name, date: updated.appointment_date, time: updated.appointment_time } });
      } catch (e) {
        console.error('Failed to notify receptionists:', e);
      }

      return updated;
    } catch (error) {
      throw error;
    }
  }

  // Confirm by token (email link)
  async confirmByToken(id, token) {
    try {
      if (!token) throw new Error('Missing token');
      const appointment = await prisma.appointments.findUnique({ where: { id: parseInt(id) }, include: { patient: { include: { user: true } }, doctor: { include: { user: true } } } });
      if (!appointment) throw new Error('Appointment not found');
      if (!appointment.confirmation_token || appointment.confirmation_token !== token) {
        throw new Error('Invalid or expired token');
      }

      const updated = await prisma.appointments.update({ where: { id: parseInt(id) }, data: { status: 'CONFIRMED', patient_confirmed: true, patient_confirmed_at: new Date(), confirmation_token: null, updated_at: new Date() }, include: { patient: { include: { user: true } }, doctor: { include: { user: true } } } });

      // Send broadcast to reception
      try {
        await notificationsService.broadcastNotification({ role_name: 'Receptionist', type: 'patient_confirmed', payload: { appointment_id: updated.id, patient_name: updated.patient?.full_name || updated.patient?.user?.full_name, doctor_name: updated.doctor?.user?.full_name, date: updated.appointment_date, time: updated.appointment_time } });
      } catch (e) {
        console.error('Failed to notify receptionists after token confirm:', e);
      }

      // audit log
      try {
        await prisma.audit_logs.create({ data: { user_id: updated.patient?.user_id || null, action: 'appointment:confirmed_via_email', meta: { appointment_id: updated.id } } });
      } catch (e) {
        console.error('Failed to write audit log for token confirm:', e);
      }

      // Send confirmation email (non-blocking)
      (async () => {
        try {
          const patientEmail = updated.patient?.user?.email || updated.patient?.email;
          if (patientEmail) {
            await emailService.sendAppointmentConfirmation(updated, patientEmail);
          }
        } catch (e) {
          console.error('Failed to send confirmation email after token confirm:', e);
        }
      })();

      return this.formatAppointment(updated);
    } catch (error) {
      throw error;
    }
  }

  async checkInAppointment(id) {
    const appointment = await prisma.appointments.findUnique({
      where: { id: parseInt(id) },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    // Allow receptionist to check-in appointments that are either CONFIRMED or still PENDING
    if (![APPOINTMENT_STATUS.CONFIRMED, APPOINTMENT_STATUS.PENDING].includes(appointment.status)) {
      throw new Error('Only confirmed or pending appointments can be checked in');
    }

    const updated = await prisma.appointments.update({
      where: { id: parseInt(id) },
      data: {
        status: APPOINTMENT_STATUS.CHECKED_IN,
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
        timeslot: true,
      },
    });

    // After check-in, create a notification for the doctor (if doctor user exists)
    try {
      const doctorUserId = updated?.doctor?.user?.id;
      if (doctorUserId) {
        await notificationsService.createNotification({
          user_id: doctorUserId,
          type: 'appointment_checkin',
          payload: {
            appointment_id: updated.id,
            patient_name: updated.patient?.user?.full_name || updated.patient?.full_name,
            appointment_date: updated.appointment_date,
            appointment_time: updated.appointment_time,
            timeslot_id: updated.timeslot_id,
          },
        });
      }
    } catch (notifyErr) {
      console.error('Failed to notify doctor on check-in:', notifyErr);
      // do not block check-in success if notification fails
    }

    return this.formatAppointment(updated);
  }

  async startAppointment(id) {
    const appointment = await prisma.appointments.findUnique({
      where: { id: parseInt(id) },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    if (appointment.status !== APPOINTMENT_STATUS.CHECKED_IN) {
      throw new Error('Patient must be checked in first');
    }

    const updated = await prisma.appointments.update({
      where: { id: parseInt(id) },
      data: {
        status: APPOINTMENT_STATUS.IN_PROGRESS,
        updated_at: new Date(),
      },
    });

    return this.formatAppointment(updated);
  }

  async completeAppointment(id) {
    const appointment = await prisma.appointments.findUnique({
      where: { id: parseInt(id) },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    if (appointment.status !== APPOINTMENT_STATUS.IN_PROGRESS) {
      throw new Error('Only in-progress appointments can be completed');
    }

    const updated = await prisma.appointments.update({
      where: { id: parseInt(id) },
      data: {
        status: APPOINTMENT_STATUS.COMPLETED,
        updated_at: new Date(),
      },
    });

    return this.formatAppointment(updated);
  }

  async markNoShow(id) {
    const appointment = await prisma.appointments.findUnique({
      where: { id: parseInt(id) },
    });

    if (!appointment) {
      throw new Error('Appointment not found');
    }

    if (![APPOINTMENT_STATUS.CONFIRMED, APPOINTMENT_STATUS.CHECKED_IN].includes(appointment.status)) {
      throw new Error('Only confirmed or checked-in appointments can be marked as no-show');
    }

    const updated = await prisma.appointments.update({
      where: { id: parseInt(id) },
      data: {
        status: APPOINTMENT_STATUS.NO_SHOW,
        updated_at: new Date(),
      },
    });

    return this.formatAppointment(updated);
  }

  async deleteAppointment(id, user) {
    // Only allow certain roles to permanently delete; controller route enforces authorize, but double-check
    const appointment = await prisma.appointments.findUnique({ where: { id: parseInt(id) } });
    if (!appointment) throw new Error('Appointment not found');

    // Use transaction: delete appointment and decrement timeslot booked_count if applicable
    const result = await prisma.$transaction(async (tx) => {
      let updatedTimeslot = null;
      if (appointment.timeslot_id) {
        // decrement booked_count but don't go below 0
        updatedTimeslot = await tx.timeslots.updateMany({
          where: { id: appointment.timeslot_id, booked_count: { gt: 0 } },
          data: { booked_count: { decrement: 1 } },
        });
      }

      await tx.appointments.delete({ where: { id: parseInt(id) } });

      return { deletedId: parseInt(id) };
    });

    return result;
  }

  async rescheduleAppointment(appointmentId, newTimeslotId, user) {
    try {
      const appointment = await prisma.appointments.findUnique({
        where: { id: parseInt(appointmentId) }
      });

      if (!appointment) {
        throw new Error('Appointment not found');
      }

      // Authorization check
      if (user.role.name === ROLES.PATIENT) {
        const patient = await prisma.patients.findFirst({
          where: { user_id: user.id }
        });
        if (!patient || patient.id !== appointment.patient_id) {
          throw new Error('Unauthorized: cannot reschedule this appointment');
        }
      }

      if (![APPOINTMENT_STATUS.PENDING, APPOINTMENT_STATUS.CONFIRMED].includes(appointment.status)) {
        throw new Error('Cannot reschedule appointment with status: ' + appointment.status);
      }

      const newTimeslot = await prisma.timeslots.findUnique({
        where: { id: parseInt(newTimeslotId) }
      });

      if (!newTimeslot) {
        throw new Error('Timeslot not found');
      }

      if (!newTimeslot.is_active) {
        throw new Error('Timeslot is not available');
      }

      if (newTimeslot.booked_count >= newTimeslot.max_patients) {
        throw new Error('Timeslot is fully booked');
      }

      // Use transaction to prevent double-booking
      const rescheduled = await prisma.$transaction(async (tx) => {
        // Release old timeslot
        if (appointment.timeslot_id) {
          await tx.timeslots.update({
            where: { id: appointment.timeslot_id },
            data: {
              booked_count: {
                decrement: 1
              }
            }
          });
        }

        // Book new timeslot
        await tx.timeslots.update({
          where: { id: newTimeslot.id },
          data: {
            booked_count: {
              increment: 1
            }
          }
        });

        // Update appointment
        const updated = await tx.appointments.update({
          where: { id: parseInt(appointmentId) },
          data: {
            timeslot_id: newTimeslot.id,
            appointment_date: newTimeslot.date,
            appointment_time: newTimeslot.start_time,
            updated_at: new Date()
          },
          include: {
            patient: {
              include: {
                user: {
                  select: {
                    full_name: true,
                    email: true,
                    phone: true
                  }
                }
              }
            },
            doctor: {
              include: {
                user: {
                  select: {
                    full_name: true
                  }
                }
              }
            },
            timeslot: true
          }
        });

        return this.formatAppointment(updated);
      });

      return this.formatAppointment(rescheduled);
    } catch (error) {
      throw error;
    }
  }

  async getAppointmentHistory(user) {
    try {
      // Get patient_id from current user
      let patient_id;
      if (user.role.name === ROLES.PATIENT) {
        const patient = await prisma.patients.findFirst({
          where: { user_id: user.id },
        });
        if (!patient) {
          throw new Error('Patient profile not found');
        }
        patient_id = patient.id;
      } else {
        throw new Error('Only patients can view appointment history');
      }

      // Get current datetime
      const now = new Date();

      // Fetch all appointments for this patient, sorted by date desc
      const appointments = await prisma.appointments.findMany({
        where: {
          patient_id: patient_id,
        },
        include: {
          patient: {
            select: {
              id: true,
              full_name: true,
              phone: true,
              email: true,
              dob: true,
              gender: true,
              blood_type: true,
              allergies: true,
              address: true,
              occupation: true,
              id_type: true,
              id_number: true,
              id_issue_date: true,
              id_issue_place: true,
              nationality: true,
              ethnicity: true,
              zalo: true,
              user: {
                select: {
                  id: true,
                  full_name: true,
                  phone: true,
                  email: true,
                },
              },
              medical_records: {
                orderBy: { created_at: 'desc' },
                select: {
                  id: true,
                  diagnosis: true,
                  notes: true,
                  created_at: true,
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
              doctorSpecialties: {
                include: {
                  specialty: {
                    select: {
                      id: true,
                      name: true,
                    },
                  },
                },
              },
            },
          },
          timeslot: true,
        },
        orderBy: [
          { appointment_date: 'desc' },
          { appointment_time: 'desc' },
        ],
      });

      return {
        appointments: this.formatAppointments(appointments),
        total: appointments.length,
      };
    } catch (error) {
      throw error;
    }
  }

  async getAllAppointmentsByPatient(user) {
    try {
      // Get patient by user_id
      const patient = await prisma.patients.findFirst({
        where: { user_id: user.id },
      });

      if (!patient) {
        return {
          appointments: [],
          total: 0,
        };
      }

      // Get ALL appointments for this patient (no pagination, no status filter)
      const appointments = await prisma.appointments.findMany({
        where: {
          patient_id: patient.id,
        },
        include: {
          patient: {
            include: {
              user: {
                select: {
                  id: true,
                  full_name: true,
                  phone: true,
                  email: true,
                },
              },
              medical_records: true,
            },
          },
          doctor: {
            include: {
              user: {
                select: {
                  id: true,
                  full_name: true,
                  phone: true,
                  email: true,
                },
              },
              doctorSpecialties: {
                include: {
                  specialty: true,
                },
              },
            },
          },
          timeslot: true,
        },
        orderBy: [
          { appointment_date: 'desc' },
          { appointment_time: 'desc' },
        ],
      });

      // Normalize each appointment
      const normalized = appointments.map((apt) => {
        try {
          const dateObj = apt.appointment_date instanceof Date ? apt.appointment_date : new Date(apt.appointment_date);
          const timeSource = apt.appointment_time || (apt.timeslot && apt.timeslot.start_time) || null;

          let hours = 0, minutes = 0, seconds = 0;
          if (timeSource) {
            const t = timeSource instanceof Date ? timeSource : new Date(timeSource);
            if (!isNaN(t.getTime())) {
              hours = t.getUTCHours();
              minutes = t.getUTCMinutes();
              seconds = t.getUTCSeconds();
            }
          }

          const y = dateObj.getUTCFullYear();
          const mo = dateObj.getUTCMonth();
          const d = dateObj.getUTCDate();
          const dt = new Date(Date.UTC(y, mo, d, hours, minutes, seconds));

          return {
            ...apt,
            appointment_datetime: dt.toISOString(),
          };
        } catch {
          return apt;
        }
      });

      return {
        appointments: normalized,
        total: appointments.length,
      };
    } catch (error) {
      throw error;
    }
  }

  // Return combined view for patient: upcoming appointments and history
  async getMyAppointmentsForPatient(user, options = {}) {
    try {
      const { page = 1, limit = 10 } = options;

      // Get all patient profiles related to this user (either owned or linked)
      const patients = await prisma.patients.findMany({
        where: {
          OR: [
            { user_id: user.id },
            { owner_user_id: user.id }
          ]
        }
      });

      if (!patients || patients.length === 0) return { upcoming: [], history: [], totalUpcoming: 0, totalHistory: 0 };

      const patientIds = patients.map(p => p.id);

      // Fetch all appointments for these patient ids with relations
      const appointments = await prisma.appointments.findMany({
        where: { patient_id: { in: patientIds } },
        include: {
          patient: {
            include: {
              user: {
                select: { id: true, full_name: true, phone: true, email: true },
              },
            },
          },
          doctor: {
            include: {
              user: { select: { id: true, full_name: true, phone: true, email: true } },
              doctorSpecialties: { include: { specialty: true } },
            },
          },
          timeslot: true,
        },
        orderBy: [
          { appointment_date: 'desc' },
          { appointment_time: 'desc' },
        ],
      });

      // Partition into upcoming and history using local date (midnight)
      const today = new Date();
      today.setHours(0, 0, 0, 0);

      const upcoming = [];
      const history = [];

      for (const apt of appointments) {
        const aptDate = apt.appointment_date instanceof Date ? new Date(apt.appointment_date) : new Date(apt.appointment_date);
        // Compare date only
        const aptDay = new Date(aptDate);
        aptDay.setHours(0, 0, 0, 0);

        // treat CANCELLED / NO_SHOW as history
        const isHistoryStatus = ['CANCELLED', 'NO_SHOW', 'COMPLETED'].includes(apt.status);

        if (aptDay < today || isHistoryStatus) {
          history.push(apt);
        } else {
          upcoming.push(apt);
        }
      }

      // Optionally apply pagination to upcoming list
      const start = (page - 1) * limit;
      const pagedUpcoming = upcoming.slice(start, start + limit);

      return {
        upcoming: this.formatAppointments(pagedUpcoming),
        history: this.formatAppointments(history),
        totalUpcoming: upcoming.length,
        totalHistory: history.length,
        page,
        limit,
      };
    } catch (error) {
      throw error;
    }
  }
}

module.exports = new AppointmentService();