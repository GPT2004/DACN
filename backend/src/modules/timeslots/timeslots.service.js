/* eslint-disable no-console */
const prisma = require('../../config/database');

class TimeslotsService {
  async getAllTimeslots(filters = {}, pagination = { page: 1, limit: 50 }) {
    try {
      const { page, limit } = pagination;
      const skip = (page - 1) * limit;

      const where = {};

      if (filters.doctor_id) {
        where.doctor_id = filters.doctor_id;
      }

      if (filters.date) {
        where.date = new Date(filters.date);
      }

      if (filters.available) {
        where.is_active = true;
        where.booked_count = {
          lt: prisma.timeslots.fields.max_patients
        };
      }

      const [total, timeslots] = await Promise.all([
        prisma.timeslots.count({ where }),
        prisma.timeslots.findMany({
          where,
          skip,
          take: limit,
          include: {
            doctors: {
              include: {
                users: {
                  select: {
                    full_name: true,
                    phone: true
                  }
                }
              }
            },
            _count: {
              select: {
                appointments: true
              }
            }
          },
          orderBy: [
            { date: 'asc' },
            { start_time: 'asc' }
          ]
        })
      ]);

      const formatTimeField = (timeVal) => {
        if (!timeVal) return null;
        const str = String(timeVal);
        const match = str.match(/^(\d{2}):(\d{2})/);
        return match ? `${match[1]}:${match[2]}` : str.substring(0, 5);
      };

      return {
        ...updatedTimeslot,
        start_time: formatTimeField(updatedTimeslot.start_time),
        end_time: formatTimeField(updatedTimeslot.end_time)
      };
      return {
        timeslots: formattedTimeslots,
        total,
        page,
        limit,
        totalPages: Math.ceil(total / limit)
      };
    } catch (error) {
      console.error('Error in getAllTimeslots:', error);
      throw error;
    }
  }

  async getTimeslotById(id) {
    try {
      const timeslot = await prisma.timeslots.findUnique({
        where: { id },
        include: {
          doctors: {
            include: {
              users: {
                select: {
                  full_name: true,
                  phone: true
                }
              }
            }
          },
          appointments: {
            include: {
              patients: {
                include: {
                  users: {
                    select: {
                      full_name: true,
                      phone: true
                    }
                  }
                }
              }
            }
          }
        }
      });

      if (!timeslot) return null;

      // Helper function to format TIME fields to HH:MM string
      const formatTimeField = (timeVal) => {
        if (!timeVal) return null;
        const str = String(timeVal);
        const match = str.match(/^(\d{2}):(\d{2})/);
        return match ? `${match[1]}:${match[2]}` : str.substring(0, 5);
      };

      return {
        ...timeslot,
        start_time: formatTimeField(timeslot.start_time),
        end_time: formatTimeField(timeslot.end_time)
      };
    } catch (error) {
      console.error('Error in getTimeslotById:', error);
      throw error;
    }
  }

  async getAvailableTimeslots(doctorId, date) {
    try {
      // Compare by DATE string to avoid timezone/timestamp conversion issues
      // Expect `date` as 'YYYY-MM-DD'
      const parts = String(date).split('-').map(Number);
      if (parts.length !== 3) {
        throw new Error('Invalid date');
      }
      const [year, month, day] = parts;
      const dateStr = `${year}-${String(month).padStart(2,'0')}-${String(day).padStart(2,'0')}`;

      console.log(`🔍 [getAvailableTimeslots] Query: doctor_id=${doctorId}, date=${dateStr}`);

      // Use raw query to get TIME fields as text (HH:MM:SS) and compare date using ::date
      const timeslots = await prisma.$queryRaw`
        SELECT 
          id,
          doctor_id,
          date,
          to_char(start_time, 'HH24:MI:SS') as start_time,
          to_char(end_time, 'HH24:MI:SS') as end_time,
          max_patients,
          booked_count,
          is_active,
          created_at,
          updated_at
        FROM timeslots
        WHERE doctor_id = ${doctorId}
          AND date = ${dateStr}::date
          AND is_active = true
          AND booked_count < max_patients
        ORDER BY start_time ASC
      `;

      console.log(`✅ [getAvailableTimeslots] Found ${timeslots.length} slots`);
      if (timeslots.length === 0) {
        console.log(`⚠️  [getAvailableTimeslots] No slots found for doctor_id=${doctorId}, date=${dateStr}`);
      }

      // Return with HH:MM format (just extract first 5 chars)
      return timeslots.map(slot => ({
        ...slot,
        start_time: String(slot.start_time).substring(0, 5),
        end_time: String(slot.end_time).substring(0, 5),
        available_slots: slot.max_patients - slot.booked_count
      }));
    } catch (error) {
      console.error('❌ Error in getAvailableTimeslots:', error);
      throw error;
    }
  }

  async getDoctorTimeslots(doctorId, { startDate, endDate }) {
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);

      const timeslots = await prisma.timeslots.findMany({
        where: {
          doctor_id: doctorId,
          date: {
            gte: start,
            lte: end
          }
        },
        orderBy: [
          { date: 'asc' },
          { start_time: 'asc' }
        ]
      });

      // Helper function to format TIME fields to HH:MM string
      const formatTimeField = (timeVal) => {
        if (!timeVal) return null;
        if (timeVal instanceof Date) {
          // Treat Date as UTC time to avoid server-local timezone shifts
          const h = String(timeVal.getUTCHours()).padStart(2, '0');
          const m = String(timeVal.getUTCMinutes()).padStart(2, '0');
          return `${h}:${m}`;
        }
        if (typeof timeVal === 'string') {
          const match = timeVal.match(/(\d{2}):(\d{2})/);
          return match ? `${match[1]}:${match[2]}` : timeVal;
        }
        return String(timeVal);
      };
      // Helper to format date (YYYY-MM-DD) from Date or string
      const formatDateKey = (d) => {
        if (!d) return null;
        const dateObj = d instanceof Date ? d : new Date(d);
        if (isNaN(dateObj.getTime())) return null;
        const y = dateObj.getUTCFullYear();
        const m = String(dateObj.getUTCMonth() + 1).padStart(2, '0');
        const day = String(dateObj.getUTCDate()).padStart(2, '0');
        return `${y}-${m}-${day}`;
      };

      return timeslots.map(slot => ({
        ...slot,
        date: formatDateKey(slot.date),
        start_time: formatTimeField(slot.start_time),
        end_time: formatTimeField(slot.end_time)
      }));
    } catch (error) {
      console.error('Error in getDoctorTimeslots:', error);
      throw error;
    }
  }

  async createTimeslot(timeslotData) {
    try {
      const { doctor_id, date, start_time, end_time, max_patients = 1 } = timeslotData;

      // Check if doctor exists
      const doctor = await prisma.doctors.findUnique({
        where: { id: doctor_id }
      });

      if (!doctor) {
        throw new Error('Doctor not found');
      }

      // Check for overlapping timeslots
      const overlapping = await prisma.timeslots.findFirst({
        where: {
          doctor_id,
          date: new Date(date),
          OR: [
            {
              AND: [
                { start_time: { lte: start_time } },
                { end_time: { gt: start_time } }
              ]
            },
            {
              AND: [
                { start_time: { lt: end_time } },
                { end_time: { gte: end_time } }
              ]
            }
          ]
        }
      });

      if (overlapping) {
        throw new Error('Timeslot overlaps with existing timeslot');
      }

      const timeslot = await prisma.timeslots.create({
        data: {
          doctor_id,
          date: new Date(date),
          start_time,
          end_time,
          max_patients,
          booked_count: 0,
          is_active: true
        },
        include: {
          doctors: {
            include: {
              users: {
                select: {
                  full_name: true
                }
              }
            }
          }
        }
      });

      const formatTimeField = (timeVal) => {
        if (!timeVal) return null;
        const str = String(timeVal);
        const match = str.match(/^(\d{2}):(\d{2})/);
        return match ? `${match[1]}:${match[2]}` : str.substring(0, 5);
      };

      return {
        ...timeslot,
        start_time: formatTimeField(timeslot.start_time),
        end_time: formatTimeField(timeslot.end_time)
      };
    } catch (error) {
      console.error('Error in createTimeslot:', error);
      throw error;
    }
  }

  async createMultipleTimeslots({ doctor_id, date, start_time, end_time, duration = 20, max_patients = 1 }) {
    try {
      // Parse times
      const [startHour, startMin] = start_time.split(':').map(Number);
      const [endHour, endMin] = end_time.split(':').map(Number);

      const startMinutes = startHour * 60 + startMin;
      const endMinutes = endHour * 60 + endMin;

      const timeslots = [];
      let currentMinutes = startMinutes;

      while (currentMinutes + duration <= endMinutes) {
        const slotStartHour = Math.floor(currentMinutes / 60);
        const slotStartMin = currentMinutes % 60;
        const slotEndMinutes = currentMinutes + duration;
        const slotEndHour = Math.floor(slotEndMinutes / 60);
        const slotEndMin = slotEndMinutes % 60;

        const slotStartTime = `${String(slotStartHour).padStart(2, '0')}:${String(slotStartMin).padStart(2, '0')}`;
        const slotEndTime = `${String(slotEndHour).padStart(2, '0')}:${String(slotEndMin).padStart(2, '0')}`;

        timeslots.push({
          doctor_id,
          date: new Date(date),
          start_time: slotStartTime,
          end_time: slotEndTime,
          max_patients,
          booked_count: 0,
          is_active: true
        });

        currentMinutes += duration;
      }

      // Create all timeslots
      const created = await prisma.timeslots.createMany({
        data: timeslots,
        skipDuplicates: true
      });

      return {
        created: created.count,
        timeslots: timeslots.length
      };
    } catch (error) {
      console.error('Error in createMultipleTimeslots:', error);
      throw error;
    }
  }

  async updateTimeslot(id, updateData) {
    try {
      const timeslot = await prisma.timeslots.findUnique({
        where: { id }
      });

      if (!timeslot) {
        return null;
      }

      // Don't allow updating if there are bookings
      if (timeslot.booked_count > 0 && (updateData.date || updateData.start_time || updateData.end_time)) {
        throw new Error('Cannot modify timeslot with existing bookings');
      }

      const updatedTimeslot = await prisma.timeslots.update({
        where: { id },
        data: {
          ...updateData,
          date: updateData.date ? new Date(updateData.date) : undefined,
          updated_at: new Date()
        },
        include: {
          doctors: {
            include: {
              users: {
                select: {
                  full_name: true
                }
              }
            }
          }
        }
      });

      // Helper function to format TIME fields to HH:MM string
      const formatTimeField = (timeVal) => {
        if (!timeVal) return null;
        if (timeVal instanceof Date) {
          const h = String(timeVal.getHours()).padStart(2, '0');
          const m = String(timeVal.getMinutes()).padStart(2, '0');
          return `${h}:${m}`;
        }
        if (typeof timeVal === 'string') {
          const match = timeVal.match(/(\d{2}):(\d{2})/);
          return match ? `${match[1]}:${match[2]}` : timeVal;
        }
        return String(timeVal);
      };

      return {
        ...updatedTimeslot,
        start_time: formatTimeField(updatedTimeslot.start_time),
        end_time: formatTimeField(updatedTimeslot.end_time)
      };
    } catch (error) {
      console.error('Error in updateTimeslot:', error);
      throw error;
    }
  }

  async deleteTimeslot(id) {
    try {
      const timeslot = await prisma.timeslots.findUnique({
        where: { id }
      });

      if (!timeslot) {
        throw new Error('Timeslot not found');
      }

      // Check if timeslot has appointments
      if (timeslot.booked_count > 0) {
        throw new Error('Cannot delete timeslot with bookings');
      }

      await prisma.timeslots.delete({
        where: { id }
      });
    } catch (error) {
      console.error('Error in deleteTimeslot:', error);
      throw error;
    }
  }

  async toggleTimeslotStatus(id, isActive) {
    try {
      const timeslot = await prisma.timeslots.findUnique({
        where: { id }
      });

      if (!timeslot) {
        return null;
      }

      const updatedTimeslot = await prisma.timeslots.update({
        where: { id },
        data: {
          is_active: isActive,
          updated_at: new Date()
        }
      });

      // Helper function to format TIME fields to HH:MM string
      const formatTimeField = (timeVal) => {
        if (!timeVal) return null;
        if (timeVal instanceof Date) {
          const h = String(timeVal.getHours()).padStart(2, '0');
          const m = String(timeVal.getMinutes()).padStart(2, '0');
          return `${h}:${m}`;
        }
        if (typeof timeVal === 'string') {
          const match = timeVal.match(/(\d{2}):(\d{2})/);
          return match ? `${match[1]}:${match[2]}` : timeVal;
        }
        return String(timeVal);
      };

      return {
        ...updatedTimeslot,
        start_time: formatTimeField(updatedTimeslot.start_time),
        end_time: formatTimeField(updatedTimeslot.end_time)
      };
    } catch (error) {
      console.error('Error in toggleTimeslotStatus:', error);
      throw error;
    }
  }

  async getTimeslotAppointments(timeslotId) {
    try {
      const appointments = await prisma.appointments.findMany({
        where: {
          timeslot_id: timeslotId
        },
        include: {
          patients: {
            include: {
              users: {
                select: {
                  full_name: true,
                  phone: true,
                  email: true
                }
              }
            }
          }
        },
        orderBy: {
          created_at: 'asc'
        }
      });

      return appointments;
    } catch (error) {
      console.error('Error in getTimeslotAppointments:', error);
      throw error;
    }
  }

  async bulkUpdateTimeslots({ doctor_id, date, is_active }) {
    try {
      const where = {
        doctor_id,
        date: new Date(date)
      };

      const result = await prisma.timeslots.updateMany({
        where,
        data: {
          is_active,
          updated_at: new Date()
        }
      });

      return {
        updated: result.count
      };
    } catch (error) {
      console.error('Error in bulkUpdateTimeslots:', error);
      throw error;
    }
  }

  async checkTimeslotAvailability(timeslotId) {
    try {
      const timeslot = await prisma.timeslots.findUnique({
        where: { id: timeslotId }
      });

      if (!timeslot) {
        return null;
      }

      const isAvailable = timeslot.is_active && timeslot.booked_count < timeslot.max_patients;

      return {
        timeslot,
        isAvailable,
        availableSlots: timeslot.max_patients - timeslot.booked_count
      };
    } catch (error) {
      console.error('Error in checkTimeslotAvailability:', error);
      throw error;
    }
  }
}

module.exports = new TimeslotsService();
