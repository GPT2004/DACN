const prisma = require('../../config/database');
const { ROLES } = require('../../config/constants');

class ScheduleService {
  async createSchedule(data, user) {
    // Get doctor_id
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

    // Check for overlapping schedules
    const overlapping = await prisma.schedules.findFirst({
      where: {
        doctor_id: parseInt(doctor_id),
        date: new Date(data.date),
        OR: [
          {
            AND: [
              { start_time: { lte: data.start_time } },
              { end_time: { gt: data.start_time } },
            ],
          },
          {
            AND: [
              { start_time: { lt: data.end_time } },
              { end_time: { gte: data.end_time } },
            ],
          },
        ],
      },
    });

    if (overlapping) {
      throw new Error('Schedule overlaps with existing schedule');
    }

    // Create schedule
    const schedule = await prisma.schedules.create({
      data: {
        doctor_id: parseInt(doctor_id),
        room_id: data.room_id ? parseInt(data.room_id) : null,
        date: new Date(data.date),
        start_time: data.start_time,
        end_time: data.end_time,
        recurrent_rule: data.recurrent_rule || null,
      },
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
        room: true,
      },
    });

    // Auto-create timeslots automatically
    const slotDuration = data.slot_duration_minutes || data.slot_duration || 20;
    try {
      console.log(`📅 [SCHEDULE] Creating timeslots for schedule ID: ${schedule.id}, duration: ${slotDuration}min`);
      await this.generateTimeslots(schedule.id, slotDuration);
      console.log(`✅ [SCHEDULE] Timeslots created successfully`);
    } catch (err) {
      console.error(`❌ [SCHEDULE] Error creating timeslots:`, err.message);
      console.error(`❌ [SCHEDULE] Full error:`, err);
      console.error(`❌ [SCHEDULE] Stack:`, err.stack);
    }

    return schedule;
  }

  async generateTimeslots(schedule_id, duration_minutes = 20) {
    const schedule = await prisma.schedules.findUnique({
      where: { id: parseInt(schedule_id) },
    });

    if (!schedule) {
      throw new Error('Schedule not found');
    }

    console.log(`🔧 [GENERATETIMESLOTS] Schedule found:`, {
      doctor_id: schedule.doctor_id,
      date: schedule.date,
      start_time: schedule.start_time,
      end_time: schedule.end_time,
    });

    const timeslots = [];
    const startTime = this.timeToMinutes(schedule.start_time);
    const endTime = this.timeToMinutes(schedule.end_time);

    console.log(`⏰ [GENERATETIMESLOTS] Time range: ${startTime}min - ${endTime}min (duration: ${duration_minutes}min)`);

    for (let time = startTime; time < endTime; time += duration_minutes) {
      const start = this.minutesToTime(time);
      const end = this.minutesToTime(time + duration_minutes);

      timeslots.push({
        doctor_id: schedule.doctor_id,
        date: schedule.date,
        start_time: start,
        end_time: end,
        max_patients: 1,
        booked_count: 0,
        is_active: true,
      });
    }

    console.log(`📝 [GENERATETIMESLOTS] Generated ${timeslots.length} timeslots`);

    // Bulk create timeslots
    const result = await prisma.timeslots.createMany({
      data: timeslots,
      skipDuplicates: true,
    });

    console.log(`✅ [GENERATETIMESLOTS] Created ${result.count} timeslots in database`);

    return timeslots;
  }

  timeToMinutes(timeString) {
    let timeStr;
    
    // Handle Date object from database
    if (timeString instanceof Date) {
      const hours = String(timeString.getUTCHours()).padStart(2, '0');
      const minutes = String(timeString.getUTCMinutes()).padStart(2, '0');
      timeStr = `${hours}:${minutes}`;
    } else {
      timeStr = timeString;
    }

    const [hours, minutes] = timeStr.split(':').map(Number);
    return hours * 60 + minutes;
  }

  minutesToTime(minutes) {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    const timeStr = `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}:00.000Z`;
    // Return as Date object to match database Time(6) type
    return new Date(`1970-01-01T${timeStr}`);
  }

  async getSchedules(filters, user) {
    const { doctor_id, date, page, limit } = filters;
    const skip = (page - 1) * limit;

    const where = {};

    // Role-based filtering
    if (user.role.name === ROLES.DOCTOR) {
      const doctor = await prisma.doctors.findFirst({
        where: { user_id: user.id },
      });
      if (doctor) {
        where.doctor_id = doctor.id;
      }
    } else if (doctor_id) {
      where.doctor_id = parseInt(doctor_id);
    }

    if (date) {
      where.date = new Date(date);
    } else {
      where.date = { gte: new Date() };
    }

    const [schedules, total] = await Promise.all([
      prisma.schedules.findMany({
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
          room: true,
        },
        orderBy: [
          { date: 'asc' },
          { start_time: 'asc' },
        ],
      }),
      prisma.schedules.count({ where }),
    ]);

    return {
      schedules,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };
  }

  async getScheduleById(id) {
    const schedule = await prisma.schedules.findUnique({
      where: { id: parseInt(id) },
      include: {
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
        room: true,
      },
    });

    if (!schedule) {
      throw new Error('Schedule not found');
    }

    return schedule;
  }

  async updateSchedule(id, data) {
    const schedule = await prisma.schedules.findUnique({
      where: { id: parseInt(id) },
    });

    if (!schedule) {
      throw new Error('Schedule not found');
    }

    const updateData = {};
    if (data.room_id !== undefined) updateData.room_id = data.room_id ? parseInt(data.room_id) : null;
    if (data.date) updateData.date = new Date(data.date);
    if (data.start_time) updateData.start_time = data.start_time;
    if (data.end_time) updateData.end_time = data.end_time;
    if (data.recurrent_rule !== undefined) updateData.recurrent_rule = data.recurrent_rule;

    const updated = await prisma.schedules.update({
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
        room: true,
      },
    });

    return updated;
  }

  async deleteSchedule(id) {
    const schedule = await prisma.schedules.findUnique({
      where: { id: parseInt(id) },
    });

    if (!schedule) {
      throw new Error('Schedule not found');
    }

    // Check if there are appointments linked to this schedule's timeslots
    const hasAppointments = await prisma.appointments.findFirst({
      where: {
        timeslot: {
          doctor_id: schedule.doctor_id,
          date: schedule.date,
          start_time: { gte: schedule.start_time },
          end_time: { lte: schedule.end_time },
        },
        status: {
          notIn: ['CANCELLED', 'NO_SHOW', 'COMPLETED'],
        },
      },
    });

    if (hasAppointments) {
      throw new Error('Cannot delete schedule with active appointments');
    }

    await prisma.schedules.delete({
      where: { id: parseInt(id) },
    });

    return { message: 'Schedule deleted successfully' };
  }

  // Timeslot methods
  async getAvailableTimeslots(doctor_id, date) {
    if (!doctor_id || !date) {
      throw new Error('doctor_id and date are required');
    }

    const timeslots = await prisma.timeslots.findMany({
      where: {
        doctor_id: parseInt(doctor_id),
        date: new Date(date),
        is_active: true,
        booked_count: {
          lt: prisma.timeslots.fields.max_patients,
        },
      },
      orderBy: {
        start_time: 'asc',
      },
    });

    return timeslots;
  }

  async createTimeslot(data) {
    // Check for duplicate
    const existing = await prisma.timeslots.findFirst({
      where: {
        doctor_id: parseInt(data.doctor_id),
        date: new Date(data.date),
        start_time: data.start_time,
        end_time: data.end_time,
      },
    });

    if (existing) {
      throw new Error('Timeslot already exists');
    }

    const timeslot = await prisma.timeslots.create({
      data: {
        doctor_id: parseInt(data.doctor_id),
        date: new Date(data.date),
        start_time: data.start_time,
        end_time: data.end_time,
        max_patients: data.max_patients || 1,
        booked_count: 0,
        is_active: true,
      },
    });

    return timeslot;
  }

  async updateTimeslot(id, data) {
    const timeslot = await prisma.timeslots.findUnique({
      where: { id: parseInt(id) },
    });

    if (!timeslot) {
      throw new Error('Timeslot not found');
    }

    const updateData = {};
    if (data.max_patients !== undefined) updateData.max_patients = parseInt(data.max_patients);
    if (data.is_active !== undefined) updateData.is_active = data.is_active;
    updateData.updated_at = new Date();

    const updated = await prisma.timeslots.update({
      where: { id: parseInt(id) },
      data: updateData,
    });

    return updated;
  }

  async deleteTimeslot(id) {
    const timeslot = await prisma.timeslots.findUnique({
      where: { id: parseInt(id) },
    });

    if (!timeslot) {
      throw new Error('Timeslot not found');
    }

    if (timeslot.booked_count > 0) {
      throw new Error('Cannot delete timeslot with bookings');
    }

    await prisma.timeslots.delete({
      where: { id: parseInt(id) },
    });

    return { message: 'Timeslot deleted successfully' };
  }
}

module.exports = new ScheduleService();