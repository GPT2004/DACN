/* eslint-disable no-console */
const prisma = require('../../config/database');
const { hashPassword } = require('../../utils/bcrypt');
const { ROLES } = require('../../config/constants');

class DoctorsService {
  async getAllDoctors(filters = {}, pagination = { page: 1, limit: 10 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {};

      if (filters.specialty) {
        where.specialties = {
          has: filters.specialty // Dùng 'has' cho String[]
        };
      }

      if (filters.is_active !== undefined) {
        where.user = {
          is_active: filters.is_active
        };
      }

      if (filters.search) {
        where.user = {
          ...where.user,
          full_name: {
            contains: filters.search,
            mode: 'insensitive'
          }
        };
      }

      const [total, doctors] = await Promise.all([
        prisma.doctors.count({ where }),
        prisma.doctors.findMany({
          where,
          skip,
          take: limit,
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                email: true,
                phone: true,
                avatar_url: true,
                is_active: true
              }
            },
            _count: {
              select: {
                appointments: true
              }
            }
          },
          orderBy: {
            user: {
              full_name: 'asc'
            }
          }
        })
      ]);

      return {
        doctors,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getAllDoctors:', error);
      throw error;
    }
  }

  async getDoctorById(id, publicView = false) {
    try {
      const doctor = await prisma.doctors.findUnique({
        where: { id },
        include: {
          user: {
            select: {
              id: true,
              full_name: true,
              email: publicView ? false : true,
              phone: true,
              avatar_url: true,
              is_active: true,
              dob: publicView ? false : true
            }
          },
          _count: {
            select: {
              appointments: true,
              schedules: true
            }
          }
        }
      });

      return doctor;
    } catch (error) {
      console.error('Error in getDoctorById:', error);
      throw error;
    }
  }

  async getDoctorByUserId(userId) {
    try {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: userId },
        include: {
          user: {
            select: {
              id: true,
              full_name: true,
              email: true,
              phone: true,
              avatar_url: true,
              is_active: true,
              dob: true
            }
          },
          _count: {
            select: {
              appointments: true,
              schedules: true
            }
          }
        }
      });

      return doctor;
    } catch (error) {
      console.error('Error in getDoctorByUserId:', error);
      throw error;
    }
  }

  async getSpecialties() {
    try {
      const specialties = await prisma.specialties.findMany({
        orderBy: {
          name: 'asc'
        }
      });

      return specialties;
    } catch (error) {
      console.error('Error in getSpecialties:', error);
      throw error;
    }
  }

  async getDoctorAppointments(doctorId, filters = {}, pagination = { page: 1, limit: 20 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = { doctor_id: doctorId };

      if (filters.startDate || filters.endDate) {
        where.appointment_date = {};
        if (filters.startDate) where.appointment_date.gte = new Date(filters.startDate);
        if (filters.endDate) {
          // Adjust endDate to include the entire day (add 23:59:59.999)
          const endDate = new Date(filters.endDate);
          endDate.setHours(23, 59, 59, 999);
          where.appointment_date.lte = endDate;
        }
      }

      if (filters.status) {
        where.status = filters.status;
      }

      const [total, appointments] = await Promise.all([
        prisma.appointments.count({ where }),
        prisma.appointments.findMany({
          where,
          skip,
          take: limit,
          include: {
            patient: {
              include: {
                user: {
                  select: {
                    full_name: true,
                    phone: true
                  }
                }
              }
            },
            timeslot: true
          },
          orderBy: [
            { appointment_date: 'desc' },
            { appointment_time: 'desc' }
          ]
        })
      ]);

      return {
        appointments,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getDoctorAppointments:', error);
      throw error;
    }
  }

  async getDoctorSchedules(doctorId, startDate, endDate) {
    try {
      const where = { doctor_id: doctorId };

      if (startDate) {
        where.date = { ...where.date, gte: new Date(startDate) };
      }

      if (endDate) {
        where.date = { ...where.date, lte: new Date(endDate) };
      }

      const schedules = await prisma.schedules.findMany({
        where,
        include: {
          room: true
        },
        orderBy: [
          { date: 'asc' },
          { start_time: 'asc' }
        ]
      });

      // Normalize time fields to HH:MM format for consistent API response
      return schedules.map(s => {
        const dateStr = s.date instanceof Date 
          ? `${s.date.getFullYear()}-${String(s.date.getMonth() + 1).padStart(2, '0')}-${String(s.date.getDate()).padStart(2, '0')}`
          : s.date;
        
        return {
          ...s,
          start_time: this.formatTimeFromDate(s.start_time),
          end_time: this.formatTimeFromDate(s.end_time),
          date: dateStr,
          room: s.room || null,
          room_id: undefined
        };
      });
    } catch (error) {
      console.error('Error in getDoctorSchedules:', error);
      throw error;
    }
  }

  // Helper to format Time from DateTime object (1970-01-01T08:30:00Z -> "08:30")
  formatTimeFromDate(dateObj) {
    if (!dateObj) return null;
    const hours = String(dateObj.getUTCHours()).padStart(2, '0');
    const minutes = String(dateObj.getUTCMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
  }

  async createSchedule(doctorId, scheduleData, currentUser = null) {
    try {
      let {
        date,
        start_time,
        end_time,
        slot_duration_minutes = 20,
        capacity = 1,
        room_id = null
      } = scheduleData;

      // Defensive coercion: ensure numeric fields are correct types
      room_id = room_id ? parseInt(String(room_id), 10) : null;
      slot_duration_minutes = Number(slot_duration_minutes) || 20;
      capacity = Number(capacity) || 1;

      console.log('🔢 [PARAMS] Received slot_duration_minutes:', slot_duration_minutes, 'capacity:', capacity, 'date:', date, 'start:', start_time, 'end:', end_time);

      // Permission check: Doctor can only create schedule for themselves
      if (currentUser?.role_name === 'Doctor') {
        const doctorRecord = await prisma.doctors.findUnique({
          where: { user_id: currentUser.id }
        });
        if (!doctorRecord || doctorRecord.id !== doctorId) {
          throw new Error('Doctors can only create schedules for themselves');
        }
      }
      // Receptionist and Admin can create for any doctor

      // validate doctor exists
      const doctor = await prisma.doctors.findUnique({ where: { id: doctorId } });
      if (!doctor) {
        throw new Error('Doctor not found');
      }

      // If a room_id was provided, ensure it exists (avoid FK violation)
      if (room_id) {
        const roomRecord = await prisma.rooms.findUnique({ where: { id: room_id } });
        if (!roomRecord) {
          const err = new Error(`Room with id ${room_id} not found`);
          err.statusCode = 400;
          throw err;
        }
      } else {
        // If no room_id provided and the current user is the doctor, try to find the doctor's assigned room
        if (currentUser?.role_name === 'Doctor') {
          try {
            const doctorRecord = await prisma.doctors.findUnique({ where: { user_id: currentUser.id } });
            if (doctorRecord) {
              const assignedRoom = await prisma.rooms.findFirst({ where: { doctor_id: doctorRecord.id } });
              if (assignedRoom) {
                room_id = assignedRoom.id;
              }
            }
          } catch (e) {
            // ignore lookup errors and proceed with room_id = null
          }
        }
      }

      // Normalize times: remove seconds if present (HH:MM:SS -> HH:MM)
      start_time = start_time.slice(0, 5);
      end_time = end_time.slice(0, 5);

      // CRITICAL: Convert date to YYYY-MM-DD string if it's a Date object
      // Prisma schema validation may convert it, so force it back to string
      let requestDate = date;
      if (date instanceof Date) {
        requestDate = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
        console.log('🔄 [DATE CONVERSION] Converted Date object to string:', requestDate);
      }
      date = requestDate;
      console.log('✅ [DATE FINAL] Final date value:', date, 'Type:', typeof date);

      // Normalize date and times for comparison
      const scheduleDate = new Date(date);
      scheduleDate.setHours(0,0,0,0);
      const today = new Date();
      today.setHours(0,0,0,0);

      // Disallow creating schedules in the past (date before today)
      if (scheduleDate < today) {
        throw new Error('Không thể tạo lịch cho ngày trong quá khứ');
      }

      // If creating schedule for today, ensure start_time is not in the past
      const normalizedStart = start_time ? start_time.slice(0,5) : null;
      if (scheduleDate.getTime() === today.getTime() && normalizedStart) {
        const [sh, sm] = normalizedStart.split(':').map(Number);
        const startDateTimeLocal = new Date(date);
        startDateTimeLocal.setHours(sh, sm, 0, 0);
        const now = new Date();
        if (startDateTimeLocal < now) {
          throw new Error('Không thể tạo lịch bắt đầu trong quá khứ cho ngày hôm nay');
        }
      }

      // Check for duplicate schedule (same date, start_time, end_time for same doctor)
      const existingSchedule = await prisma.schedules.findFirst({
        where: {
          doctor_id: doctorId,
          date: new Date(date),
          start_time: new Date(`1970-01-01T${start_time}:00Z`),
          end_time: new Date(`1970-01-01T${end_time}:00Z`)
        }
      });
      if (existingSchedule) {
        throw new Error('Schedule with same date and time already exists for this doctor');
      }

      // Check for overlapping schedules
      const startDateTime = new Date(`1970-01-01T${start_time}:00Z`);
      const endDateTime = new Date(`1970-01-01T${end_time}:00Z`);
      
      const overlappingSchedule = await prisma.schedules.findFirst({
        where: {
          doctor_id: doctorId,
          date: new Date(date),
          OR: [
            {
              AND: [
                { start_time: { lte: startDateTime } },
                { end_time: { gt: startDateTime } }
              ]
            },
            {
              AND: [
                { start_time: { lt: endDateTime } },
                { end_time: { gte: endDateTime } }
              ]
            }
          ]
        }
      });
      
      if (overlappingSchedule) {
        throw new Error(`Schedule overlaps with existing schedule from ${overlappingSchedule.start_time} to ${overlappingSchedule.end_time}`);
      }

      // Validate time format and start_time < end_time
      const timeRegex = /^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/;
      if (!timeRegex.test(start_time) || !timeRegex.test(end_time)) {
        throw new Error('Invalid time format. Use HH:MM format (e.g., 09:00)');
      }

      const [startH, startM] = start_time.split(':').map(Number);
      const [endH, endM] = end_time.split(':').map(Number);
      const startMinutes = startH * 60 + startM;
      const endMinutes = endH * 60 + endM;

      console.log('⏰ [TIME CALC] startMinutes:', startMinutes, 'endMinutes:', endMinutes, 'duration:', slot_duration_minutes);

      if (startMinutes >= endMinutes) {
        throw new Error('Start time must be before end time');
      }

      // Convert time strings to DateTime objects for Prisma (using epoch date as base)
      // Note: startDateTime, endDateTime already defined above in overlap check

      // create schedule
      // CRITICAL: Parse date string to UTC midnight
      // "2025-11-26" -> new Date("2025-11-26T00:00:00Z") = UTC midnight for that date
      const schedCreationDate = new Date(`${date}T00:00:00Z`);
      
      console.log('📅 [SCHEDULE DATE] Input date:', date, 'Parsed UTC:', schedCreationDate.toISOString());
      
      const schedule = await prisma.schedules.create({
        data: {
          doctor_id: doctorId,
          room_id: room_id,
          date: schedCreationDate,
          start_time: startDateTime,
          end_time: endDateTime,
          created_at: new Date()
        }
      });

      console.log(`📅 [SCHEDULE CREATED] ID: ${schedule.id}, Doctor: ${doctorId}, Date: ${date}, Time: ${start_time}-${end_time}`);

      // generate timeslots for the schedule date
      const slots = [];
      const pad = (n) => (n < 10 ? '0' + n : '' + n);

      console.log('🔍 [SLOT GENERATION] Input date param:', date, 'Type:', typeof date, 'String:', String(date));

      let current = startMinutes;
      let slotCount = 0;
      while (current < endMinutes) {
        const next = current + slot_duration_minutes;
        if (next > endMinutes) break;

        const currentH = Math.floor(current / 60);
        const currentM = current % 60;
        const nextH = Math.floor(next / 60);
        const nextM = next % 60;

        const slotStartTime = `${pad(currentH)}:${pad(currentM)}`;
        const slotEndTime = `${pad(nextH)}:${pad(nextM)}`;

        // Create ISO-8601 DateTime for Prisma (1970-01-01T08:00:00Z format)
        const startTimeISO = `1970-01-01T${slotStartTime}:00Z`;
        const endTimeISO = `1970-01-01T${slotEndTime}:00Z`;

        // Create date object - parse as UTC midnight
        // CRITICAL: "2025-11-26" -> new Date("2025-11-26T00:00:00Z") = UTC midnight for that date
        const dateStr = String(date);
        const adjustedTsDate = new Date(`${dateStr}T00:00:00Z`);
        
        slots.push({
          doctor_id: doctorId,
          date: adjustedTsDate,
          start_time: new Date(startTimeISO),
          end_time: new Date(endTimeISO),
          max_patients: capacity,
          booked_count: 0,
          is_active: true
        });
        slotCount++;
        current = next;
      }

      console.log('🔄 [TIMESLOTS GENERATED] Total: ' + slotCount + ' slots (' + slot_duration_minutes + ' min each, ' + startMinutes + 'min - ' + endMinutes + 'min)');

      let actualCreatedCount = 0;
      if (slots.length > 0) {
        // Log sample slot data
        try {
          console.log('📋 [SAMPLE SLOT]', {
            doctor_id: slots[0].doctor_id,
            date: String(slots[0].date),
            start_time: slots[0].start_time,
            end_time: slots[0].end_time,
            max_patients: slots[0].max_patients
          });
        } catch (logErr) {
          console.error('❌ [SAMPLE SLOT LOG ERROR]', logErr.message);
        }
        
        // First, check existing timeslots for this date
        try {
          // Use same local date parsing as slot creation
          const [year, month, day] = String(date).split('-').map(Number);
          const deleteDate = new Date(year, month - 1, day, 0, 0, 0, 0);
          
          console.log('🔍 [CHECK CLEANUP] Looking for existing timeslots:', {
            doctor_id: doctorId,
            date_obj: String(deleteDate)
          });
          
          const existingCount = await prisma.timeslots.count({
            where: {
              doctor_id: doctorId,
              date: deleteDate
            }
          });
          console.log('🔍 [FOUND]', existingCount, 'existing timeslots');
          
          if (existingCount > 0) {
            const deleteResult = await prisma.timeslots.deleteMany({
              where: {
                doctor_id: doctorId,
                date: deleteDate
              }
            });
            console.log('🗑️  [CLEANUP] Deleted', deleteResult.count, 'timeslots for doctor', doctorId, 'on', date);
          }
        } catch (deleteErr) {
          console.error('❌ [DELETE ERROR] Failed to delete existing timeslots:', deleteErr.message);
        }

        // Create timeslots one by one with detailed logging
        console.log('🚀 [CREATE START] Creating', slots.length, 'timeslots...');
        for (let i = 0; i < slots.length; i++) {
          const s = slots[i];
          try {
            const created = await prisma.timeslots.create({ 
              data: {
                doctor_id: s.doctor_id,
                date: s.date,
                start_time: s.start_time,
                end_time: s.end_time,
                max_patients: s.max_patients || 1,
                booked_count: 0,
                is_active: true
              }
            });
            console.log('✅ [TIMESLOT #' + (i + 1) + '] Created ID:', created.id);
            actualCreatedCount++;
          } catch (e) {
            console.error('❌ [TIMESLOT #' + (i + 1) + ' FAILED]', e.message, {
              doctor_id: s.doctor_id,
              date: String(s.date),
              start_time: String(s.start_time),
              end_time: String(s.end_time)
            });
          }
        }
        console.log(`✅ [TIMESLOTS COMPLETE] Created ${actualCreatedCount}/${slots.length} timeslots successfully`);
      }

      // Fetch room data if room_id exists
      let room = null;
      if (schedule.room_id) {
        room = await prisma.rooms.findUnique({ where: { id: schedule.room_id } });
      }

      // Return normalized schedule with timeslot info
      const scheduleDateStr = schedule.date instanceof Date 
        ? `${schedule.date.getFullYear()}-${String(schedule.date.getMonth() + 1).padStart(2, '0')}-${String(schedule.date.getDate()).padStart(2, '0')}`
        : schedule.date;
      
      const responseData = {
        ...schedule,
        start_time: this.formatTimeFromDate(schedule.start_time),
        end_time: this.formatTimeFromDate(schedule.end_time),
        date: scheduleDateStr,
        room: room,
        room_id: undefined,
        timeslots_created: actualCreatedCount,
        message: `✅ Schedule created successfully with ${actualCreatedCount} timeslots`
      };

      console.log(`📋 [RESPONSE] Schedule creation completed: ${actualCreatedCount} timeslots saved`);
      return responseData;
    } catch (error) {
      console.error('Error in createSchedule:', error);
      throw error;
    }
  }

  async getDoctorPatients(doctorId) {
    try {
      const appointments = await prisma.appointments.findMany({
        where: {
          doctor_id: doctorId,
          status: 'COMPLETED'
        },
        select: {
          patient_id: true
        },
        distinct: ['patient_id']
      });

      const patientIds = appointments.map(a => a.patient_id);

      const patients = await prisma.patients.findMany({
        where: {
          id: { in: patientIds }
        },
        include: {
          user: {
            select: {
              full_name: true,
              phone: true,
              email: true,
              dob: true
            }
          },
          _count: {
            select: {
              appointments: {
                where: {
                  doctor_id: doctorId
                }
              }
            }
          }
        }
      });

      return patients;
    } catch (error) {
      console.error('Error in getDoctorPatients:', error);
      throw error;
    }
  }

  async getDoctorStats(doctorId) {
    try {
      const [
        totalAppointments,
        completedAppointments,
        upcomingAppointments,
        totalPatients,
        totalRevenue
      ] = await Promise.all([
        prisma.appointments.count({
          where: { doctor_id: doctorId }
        }),
        prisma.appointments.count({
          where: {
            doctor_id: doctorId,
            status: 'COMPLETED'
          }
        }),
        prisma.appointments.count({
          where: {
            doctor_id: doctorId,
            appointment_date: { gte: new Date() },
            status: { in: ['PENDING', 'CONFIRMED', 'CHECKED_IN'] }
          }
        }),
        prisma.appointments.findMany({
          where: {
            doctor_id: doctorId,
            status: 'COMPLETED'
          },
          select: { patient_id: true },
          distinct: ['patient_id']
        }),
        // Get invoices for appointments of this doctor
        prisma.invoices.aggregate({
          where: {
            appointment: {
              doctor_id: doctorId
            },
            status: 'PAID'
          },
          _sum: {
            total: true
          }
        })
      ]);

      return {
        total_appointments: totalAppointments,
        completed_appointments: completedAppointments,
        upcoming_appointments: upcomingAppointments,
        total_patients: totalPatients.length,
        total_revenue: totalRevenue._sum.total || 0
      };
    } catch (error) {
      console.error('Error in getDoctorStats:', error);
      throw error;
    }
  }

  async createDoctor(doctorData) {
  try {
    const {
      email, password, full_name, phone, dob,
      license_number, specialties, bio, consultation_fee
    } = doctorData;

    // XỬ LÝ SPECIALTIES
    let specialtiesArray = [];
    if (typeof specialties === 'string') {
      specialtiesArray = specialties.split(',').map(s => s.trim()).filter(Boolean);
    } else if (Array.isArray(specialties)) {
      specialtiesArray = specialties;
    }

    // KIỂM TRA EMAIL TRÙNG
    const existingUser = await prisma.users.findUnique({ where: { email } });
    if (existingUser) {
      throw new Error('Email already registered');
    }

    // KIỂM TRA ROLE DOCTOR
    if (!prisma?.roles?.findUnique) {
      throw new Error('Prisma client not initialized');
    }

    const doctorRole = await prisma.roles.findUnique({
      where: { name: ROLES.DOCTOR }
    });

    if (!doctorRole) {
      throw new Error('Doctor role not found in database');
    }

    const hashedPassword = await hashPassword(password);

    const result = await prisma.$transaction(async (tx) => {
      const user = await tx.users.create({
        data: {
          email,
          password: hashedPassword,
          full_name,
          phone,
          dob: dob ? new Date(dob) : null,
          role_id: doctorRole.id,
          is_active: true
        }
      });

      const doctor = await tx.doctors.create({
        data: {
          user_id: user.id,
          license_number,
          specialties: specialtiesArray,
          bio,
          consultation_fee: consultation_fee || 0,
          rating: 0
        },
        include: {
          user: {  // ← ĐÃ SỬA TỪ `users` → `user`
            select: {
              id: true,
              full_name: true,
              email: true,
              phone: true
            }
          }
        }
      });

      return doctor;
    });

    return result;
  } catch (error) {
    console.error('Error in createDoctor:', error);
    throw error;
  }
}

  async updateDoctor(id, updateData) {
    try {
      const doctor = await prisma.doctors.findUnique({
        where: { id },
        include: { user: true }
      });

      if (!doctor) {
        return null;
      }

      // Xử lý specialty/specialties - support cả 2 field từ FE
      let specialtiesArray = doctor.specialties;
      
      // Support both 'specialty' (singular) và 'specialties' (plural)
      const specialtyInput = updateData.specialty || updateData.specialties;
      
      if (specialtyInput) {
        if (typeof specialtyInput === 'string') {
          specialtiesArray = specialtyInput
            .split(',')
            .map(s => s.trim())
            .filter(Boolean);
        } else if (Array.isArray(specialtyInput)) {
          specialtiesArray = specialtyInput;
        }
      }

      const result = await prisma.$transaction(async (tx) => {
        // Update user info
        const userUpdateData = {};
        if (updateData.full_name) userUpdateData.full_name = updateData.full_name;
        if (updateData.phone) userUpdateData.phone = updateData.phone;
        if (updateData.email) userUpdateData.email = updateData.email;
        if (updateData.dob) userUpdateData.dob = new Date(updateData.dob);

        if (Object.keys(userUpdateData).length > 0) {
          await tx.users.update({
            where: { id: doctor.user_id },
            data: userUpdateData
          });
        }

        // Update doctor info
        const doctorUpdateData = {};
        if (updateData.license_number) doctorUpdateData.license_number = updateData.license_number;
        if (specialtyInput) doctorUpdateData.specialties = specialtiesArray;
        if (updateData.bio !== undefined) doctorUpdateData.bio = updateData.bio;
        if (updateData.consultation_fee !== undefined) doctorUpdateData.consultation_fee = updateData.consultation_fee;

        const updatedDoctor = await tx.doctors.update({
          where: { id },
          data: doctorUpdateData,
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                email: true,
                phone: true,
                is_active: true
              }
            }
          }
        });

        return updatedDoctor;
      });

      return result;
    } catch (error) {
      console.error('Error in updateDoctor:', error);
      throw error;
    }
  }

  async updateDoctorProfile(userId, updateData) {
    try {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: userId }
      });

      if (!doctor) {
        return null;
      }

      let specialtiesArray = doctor.specialties;
      if (updateData.specialties) {
        if (typeof updateData.specialties === 'string') {
          specialtiesArray = updateData.specialties
            .split(',')
            .map(s => s.trim())
            .filter(Boolean);
        } else if (Array.isArray(updateData.specialties)) {
          specialtiesArray = updateData.specialties;
        }
      }

      const result = await prisma.$transaction(async (tx) => {
        // Update user info
        const userUpdateData = {};
        if (updateData.full_name) userUpdateData.full_name = updateData.full_name;
        if (updateData.phone) userUpdateData.phone = updateData.phone;
        if (updateData.avatar_url !== undefined) userUpdateData.avatar_url = updateData.avatar_url;
        if (updateData.dob) userUpdateData.dob = updateData.dob;

        if (Object.keys(userUpdateData).length > 0) {
          await tx.users.update({
            where: { id: userId },
            data: userUpdateData
          });
        }

        // Update doctor info
        const doctorUpdateData = {};
        if (updateData.bio !== undefined) doctorUpdateData.bio = updateData.bio;
        if (updateData.specialties) doctorUpdateData.specialties = specialtiesArray;
        if (updateData.consultation_fee !== undefined) doctorUpdateData.consultation_fee = updateData.consultation_fee;
        if (updateData.gender !== undefined) doctorUpdateData.gender = updateData.gender;
        if (updateData.address !== undefined) doctorUpdateData.address = updateData.address;

        const updatedDoctor = await tx.doctors.update({
          where: { id: doctor.id },
          data: doctorUpdateData,
          include: {
            user: {
              select: {
                id: true,
                full_name: true,
                email: true,
                phone: true,
                avatar_url: true,
                dob: true
              }
            }
          }
        });

        return updatedDoctor;
      });

      return result;
    } catch (error) {
      console.error('Error in updateDoctorProfile:', error);
      throw error;
    }
  }

  async deleteDoctor(id) {
    try {
      const doctor = await prisma.doctors.findUnique({
        where: { id }
      });

      if (!doctor) {
        throw new Error('Doctor not found');
      }

      const activeAppointments = await prisma.appointments.count({
        where: {
          doctor_id: id,
          appointment_date: { gte: new Date() },
          status: { in: ['PENDING', 'CONFIRMED', 'CHECKED_IN'] }
        }
      });

      if (activeAppointments > 0) {
        throw new Error('Cannot delete doctor with active appointments');
      }

      // Soft-delete associated user
      await prisma.users.update({
        where: { id: doctor.user_id },
        data: { deleted_at: new Date(), is_active: false, updated_at: new Date() }
      });
    } catch (error) {
      console.error('Error in deleteDoctor:', error);
      throw error;
    }
  }

  async toggleDoctorStatus(id) {
    try {
      const doctor = await prisma.doctors.findUnique({
        where: { id },
        include: { user: true }
      });

      if (!doctor) {
        throw new Error('Doctor not found');
      }

      await prisma.users.update({
        where: { id: doctor.user_id },
        data: {
          is_active: !doctor.user.is_active
        }
      });

      const updatedDoctor = await prisma.doctors.findUnique({
        where: { id },
        include: {
          user: {
            select: {
              id: true,
              full_name: true,
              email: true,
              is_active: true
            }
          }
        }
      });

      return updatedDoctor;
    } catch (error) {
      console.error('Error in toggleDoctorStatus:', error);
      throw error;
    }
  }

  async updateSchedule(doctorId, scheduleId, updateData) {
    try {
      const schedule = await prisma.schedules.findUnique({
        where: { id: parseInt(scheduleId) }
      });

      if (!schedule) {
        throw new Error('Schedule not found');
      }

      if (schedule.doctor_id !== doctorId) {
        throw new Error('Unauthorized: schedule does not belong to this doctor');
      }

      const updatePayload = {};

      if (updateData.date) {
        updatePayload.date = new Date(updateData.date);
      }

      if (updateData.start_time) {
        const normalizedTime = updateData.start_time.slice(0, 5);
        updatePayload.start_time = new Date(`1970-01-01T${normalizedTime}:00Z`);
      }

      if (updateData.end_time) {
        const normalizedTime = updateData.end_time.slice(0, 5);
        updatePayload.end_time = new Date(`1970-01-01T${normalizedTime}:00Z`);
      }

      if (updateData.room_id !== undefined) {
        updatePayload.room_id = updateData.room_id ? parseInt(updateData.room_id) : null;
      }

      if (updateData.recurrent_rule !== undefined) {
        updatePayload.recurrent_rule = updateData.recurrent_rule;
      }

      const updated = await prisma.schedules.update({
        where: { id: parseInt(scheduleId) },
        data: updatePayload,
        include: {
          doctor: {
            include: {
              user: {
                select: {
                  full_name: true
                }
              }
            }
          },
          room: true
        }
      });

      return updated;
    } catch (error) {
      console.error('Error in updateSchedule:', error);
      throw error;
    }
  }

  async deleteSchedule(doctorId, scheduleId) {
    try {
      const schedule = await prisma.schedules.findUnique({
        where: { id: parseInt(scheduleId) }
      });

      if (!schedule) {
        throw new Error('Schedule not found');
      }

      if (schedule.doctor_id !== doctorId) {
        throw new Error('Unauthorized: schedule does not belong to this doctor');
      }

      console.log(`📅 [DELETE SCHEDULE START] Schedule ID: ${schedule.id}, Date: ${schedule.date}, Doctor: ${schedule.doctor_id}`);

      // Check if there are active appointments for this schedule date
      const activeAppointments = await prisma.appointments.count({
        where: {
          doctor_id: schedule.doctor_id,
          appointment_date: schedule.date,
          status: {
            notIn: ['CANCELLED', 'NO_SHOW', 'COMPLETED']
          }
        }
      });

      if (activeAppointments > 0) {
        console.warn(`⚠️  [DELETE SCHEDULE] Found ${activeAppointments} active appointments - REJECTING DELETE`);
        throw new Error(`Cannot delete schedule with ${activeAppointments} active appointments`);
      }

      // DELETE ALL TIMESLOTS for this date (FORCED)
      console.log(`🗑️  [DELETE TIMESLOTS] Removing all timeslots for doctor ${schedule.doctor_id} on date ${schedule.date}`);
      
      // Log timeslots before deletion
      const timeslotsToDelete = await prisma.timeslots.findMany({
        where: {
          doctor_id: schedule.doctor_id,
          date: schedule.date
        }
      });
      console.log(`📊 [DELETE TIMESLOTS] Found ${timeslotsToDelete.length} timeslots to delete:`, timeslotsToDelete.map(t => `${t.id}: ${t.start_time}-${t.end_time}`));

      const deletedTimeslots = await prisma.timeslots.deleteMany({
        where: {
          doctor_id: schedule.doctor_id,
          date: schedule.date
        }
      });
      
      console.log(`✅ [DELETE TIMESLOTS] Successfully deleted ${deletedTimeslots.count} timeslots`);

      // DELETE THE SCHEDULE
      console.log(`🗑️  [DELETE SCHEDULE] Deleting schedule ID: ${schedule.id}`);
      const deletedSchedule = await prisma.schedules.delete({
        where: { id: parseInt(scheduleId) }
      });
      console.log(`✅ [DELETE SCHEDULE] Successfully deleted schedule ID: ${deletedSchedule.id}`);

      return { 
        message: 'Schedule and all associated timeslots deleted successfully',
        timeslots_deleted: deletedTimeslots.count,
        active_appointments: activeAppointments,
        schedule: deletedSchedule
      };
    } catch (error) {
      console.error('❌ [DELETE SCHEDULE ERROR]', error.message);
      throw error;
    }
  }

  async rescheduleAppointment(appointmentId, newTimeslotId) {
    try {
      const appointment = await prisma.appointments.findUnique({
        where: { id: parseInt(appointmentId) }
      });

      if (!appointment) {
        throw new Error('Appointment not found');
      }

      if (!['PENDING', 'CONFIRMED'].includes(appointment.status)) {
        throw new Error('Cannot reschedule appointment with status: ' + appointment.status);
      }

      const newTimeslot = await prisma.timeslots.findUnique({
        where: { id: parseInt(newTimeslotId) }
      });

      if (!newTimeslot) {
        throw new Error('Timeslot not found');
      }

      // Check if timeslot is available
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
            appointment_time: newTimeslot.start_time
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

        return updated;
      });

      return rescheduled;
    } catch (error) {
      console.error('Error in rescheduleAppointment:', error);
      throw error;
    }
  }
}

module.exports = new DoctorsService();