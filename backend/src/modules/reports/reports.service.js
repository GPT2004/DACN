/* eslint-disable no-console */
const prisma = require('../../config/database');

class ReportsService {
  async getDashboardStats() {
    try {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      // start of tomorrow for exclusive upper-bound comparisons
      const tomorrow = new Date(today);
      tomorrow.setDate(tomorrow.getDate() + 1);

      const [
        totalUsers,
        totalDoctors,
        todayAppointments,
        monthAppointments,
        todayRevenue,
        monthRevenue,
        appointmentStats,
        totalRooms
      ] = await Promise.all([
        // Total active users
        prisma.users.count({ where: { is_active: true } }).catch(() => 0),
        
        // Total active doctors (joined user must be active)
        prisma.doctors.count({ where: { user: { is_active: true } } }).catch(() => 0),
        
        // Today appointments (use range [today, tomorrow) to include all times today)
        prisma.appointments.count({
          where: {
            appointment_date: {
              gte: today,
              lt: tomorrow
            }
          }
        }).catch(() => 0),
        
        // This month appointments
        prisma.appointments.count({
          where: {
            appointment_date: {
              gte: new Date(today.getFullYear(), today.getMonth(), 1)
            }
          }
        }).catch(() => 0),
        
        // Today revenue (payments created today)
        prisma.invoices.aggregate({
          where: {
            created_at: { gte: today, lt: tomorrow },
            status: 'PAID'
          },
          _sum: { total: true }
        }).catch(() => ({ _sum: { total: null } })),
        
        // This month revenue (between start of month and start of next month)
        prisma.invoices.aggregate({
          where: {
            created_at: {
              gte: new Date(today.getFullYear(), today.getMonth(), 1),
              lt: new Date(today.getFullYear(), today.getMonth() + 1, 1)
            },
            status: 'PAID'
          },
          _sum: { total: true }
        }).catch(() => ({ _sum: { total: null } })),
        
        // Appointment status breakdown for this month
        prisma.appointments.groupBy({
          by: ['status'],
          _count: { status: true },
          where: {
            appointment_date: {
              gte: new Date(today.getFullYear(), today.getMonth(), 1),
              lt: new Date(today.getFullYear(), today.getMonth() + 1, 1)
            }
          }
        }).catch(() => []),
        
        // Total rooms
        prisma.rooms.count().catch(() => 0)
      ]);

      return {
        users: totalUsers || 0,
        doctors: totalDoctors || 0,
        todayAppointments: todayAppointments || 0,
        appointments: monthAppointments || 0,
        todayRevenue: todayRevenue?._sum?.total || 0,
        monthRevenue: monthRevenue?._sum?.total || 0,
        rooms: totalRooms || 0,
        appointmentsByStatus: (appointmentStats || []).reduce((acc, stat) => {
          acc[stat.status] = stat._count.status;
          return acc;
        }, {})
      };
    } catch (error) {
      console.error('Error in getDashboardStats:', error);
      // Return default empty stats instead of throwing
      return {
        users: 0,
        doctors: 0,
        todayAppointments: 0,
        appointments: 0,
        todayRevenue: 0,
        monthRevenue: 0,
        rooms: 0,
        appointmentsByStatus: {}
      };
    }
  }

  async getRevenueReport({ startDate, endDate, groupBy = 'day' }) {
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);
      end.setHours(23, 59, 59, 999);

      const invoices = await prisma.invoices.findMany({
        where: {
          created_at: { gte: start, lte: end },
          status: 'PAID'
        },
        select: {
          total: true,
          subtotal: true,
          tax: true,
          discount: true,
          created_at: true,
          appointments: {
            select: {
              doctors: {
                select: {
                  users: { select: { full_name: true } }
                }
              }
            }
          }
        },
        orderBy: { created_at: 'asc' }
      });

      const totalRevenue = invoices.reduce((sum, inv) => sum + inv.total, 0);
      const totalSubtotal = invoices.reduce((sum, inv) => sum + inv.subtotal, 0);
      const totalTax = invoices.reduce((sum, inv) => sum + inv.tax, 0);
      const totalDiscount = invoices.reduce((sum, inv) => sum + inv.discount, 0);

      // Group by time period
      const grouped = this.groupByPeriod(invoices, groupBy);

      return {
        summary: {
          totalRevenue,
          totalSubtotal,
          totalTax,
          totalDiscount,
          invoiceCount: invoices.length
        },
        timeline: grouped
      };
    } catch (error) {
      console.error('Error in getRevenueReport:', error);
      throw error;
    }
  }

  async getAppointmentReport({ startDate, endDate, status }) {
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);

      const where = {
        appointment_date: { gte: start, lte: end }
      };

      if (status) {
        where.status = status;
      }

      const [appointments, statusBreakdown] = await Promise.all([
        prisma.appointments.findMany({
          where,
          include: {
            doctors: {
              include: {
                users: { select: { full_name: true } }
              }
            },
            patients: {
              include: {
                users: { select: { full_name: true } }
              }
            }
          },
          orderBy: { appointment_date: 'asc' }
        }),
        
        prisma.appointments.groupBy({
          by: ['status'],
          where: { appointment_date: { gte: start, lte: end } },
          _count: { status: true }
        })
      ]);

      const byStatus = statusBreakdown.reduce((acc, item) => {
        acc[item.status] = item._count.status;
        return acc;
      }, {});

      return {
        total: appointments.length,
        byStatus,
        appointments
      };
    } catch (error) {
      console.error('Error in getAppointmentReport:', error);
      throw error;
    }
  }

  async getDoctorPerformance({ startDate, endDate, doctorId }) {
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);

      const where = {
        appointment_date: { gte: start, lte: end },
        status: 'COMPLETED'
      };

      if (doctorId) {
        where.doctor_id = doctorId;
      }

      const appointments = await prisma.appointments.groupBy({
        by: ['doctor_id'],
        where,
        _count: { id: true }
      });

      const doctorIds = appointments.map(a => a.doctor_id);

      const [doctors, revenues] = await Promise.all([
        prisma.doctors.findMany({
          where: { id: { in: doctorIds } },
          include: {
            users: { select: { full_name: true, email: true } }
          }
        }),
        
        prisma.invoices.groupBy({
          by: ['appointment_id'],
          where: {
            appointments: {
              doctor_id: { in: doctorIds },
              appointment_date: { gte: start, lte: end }
            },
            status: 'PAID'
          },
          _sum: { total: true }
        })
      ]);

      const performance = appointments.map(apt => {
        const doctor = doctors.find(d => d.id === apt.doctor_id);
        const revenue = revenues
          .filter(r => r.appointment_id)
          .reduce((sum, r) => sum + (r._sum.total || 0), 0);

        return {
          doctorId: apt.doctor_id,
          doctorName: doctor?.users?.full_name,
          totalAppointments: apt._count.id,
          totalRevenue: revenue,
          rating: doctor?.rating || 0
        };
      });

      return performance;
    } catch (error) {
      console.error('Error in getDoctorPerformance:', error);
      throw error;
    }
  }

  async getCommonDiseases({ startDate, endDate, limit = 10 }) {
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);

      const records = await prisma.medical_records.findMany({
        where: {
          created_at: { gte: start, lte: end },
          diagnosis: { not: null }
        },
        select: {
          diagnosis: true
        }
      });

      // Count diagnosis occurrences
      const diagnosisCounts = {};
      records.forEach(record => {
        const diagnosis = record.diagnosis.trim();
        if (diagnosis) {
          diagnosisCounts[diagnosis] = (diagnosisCounts[diagnosis] || 0) + 1;
        }
      });

      // Sort and limit
      const sorted = Object.entries(diagnosisCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .map(([disease, count]) => ({ disease, count }));

      return {
        diseases: sorted,
        totalRecords: records.length
      };
    } catch (error) {
      console.error('Error in getCommonDiseases:', error);
      throw error;
    }
  }

  async getStockReport() {
    try {
      const [lowStock, expiringSoon, expired] = await Promise.all([
        // Low stock (< 20 units)
        prisma.$queryRaw`
          SELECT m.id, m.name, m.code, SUM(s.quantity) as total_quantity
          FROM medicines m
          LEFT JOIN stocks s ON m.id = s.medicine_id
          WHERE s.expiry_date > CURRENT_DATE OR s.expiry_date IS NULL
          GROUP BY m.id, m.name, m.code
          HAVING SUM(s.quantity) < 20
          ORDER BY total_quantity ASC
        `,
        
        // Expiring soon (within 30 days)
        prisma.$queryRaw`
          SELECT m.name, s.batch_number, s.expiry_date, s.quantity
          FROM medicines m
          JOIN stocks s ON m.id = s.medicine_id
          WHERE s.expiry_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
          ORDER BY s.expiry_date ASC
        `,
        
        // Expired
        prisma.$queryRaw`
          SELECT m.name, s.batch_number, s.expiry_date, s.quantity
          FROM medicines m
          JOIN stocks s ON m.id = s.medicine_id
          WHERE s.expiry_date < CURRENT_DATE
          ORDER BY s.expiry_date DESC
        `
      ]);

      const totalValue = await prisma.$queryRaw`
        SELECT SUM(m.price * s.quantity) as total_value
        FROM medicines m
        JOIN stocks s ON m.id = s.medicine_id
        WHERE s.expiry_date > CURRENT_DATE OR s.expiry_date IS NULL
      `;

      return {
        lowStock,
        expiringSoon,
        expired,
        totalStockValue: totalValue[0]?.total_value || 0
      };
    } catch (error) {
      console.error('Error in getStockReport:', error);
      throw error;
    }
  }

  async getPatientReport({ startDate, endDate }) {
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);

      const [newPatients, returningPatients, ageDistribution, genderDistribution] = await Promise.all([
        // New patients
        prisma.patients.count({
          where: {
            created_at: { gte: start, lte: end }
          }
        }),
        
        // Returning patients (had appointment before this period)
        prisma.$queryRaw`
          SELECT COUNT(DISTINCT patient_id) as count
          FROM appointments
          WHERE appointment_date BETWEEN ${start} AND ${end}
          AND patient_id IN (
            SELECT DISTINCT patient_id 
            FROM appointments 
            WHERE appointment_date < ${start}
          )
        `,
        
        // Age distribution
        prisma.$queryRaw`
          SELECT 
            CASE 
              WHEN EXTRACT(YEAR FROM AGE(u.dob)) < 18 THEN '0-17'
              WHEN EXTRACT(YEAR FROM AGE(u.dob)) BETWEEN 18 AND 30 THEN '18-30'
              WHEN EXTRACT(YEAR FROM AGE(u.dob)) BETWEEN 31 AND 50 THEN '31-50'
              WHEN EXTRACT(YEAR FROM AGE(u.dob)) > 50 THEN '50+'
              ELSE 'Unknown'
            END as age_group,
            COUNT(*) as count
          FROM patients p
          JOIN users u ON p.user_id = u.id
          GROUP BY age_group
          ORDER BY age_group
        `,
        
        // Gender distribution
        prisma.patients.groupBy({
          by: ['gender'],
          _count: { gender: true }
        })
      ]);

      return {
        newPatients,
        returningPatients: returningPatients[0]?.count || 0,
        ageDistribution,
        genderDistribution: genderDistribution.reduce((acc, item) => {
          acc[item.gender || 'Unknown'] = item._count.gender;
          return acc;
        }, {})
      };
    } catch (error) {
      console.error('Error in getPatientReport:', error);
      throw error;
    }
  }

  async exportReport(type, format, params) {
    try {
      let data;

      switch (type) {
        case 'revenue':
          data = await this.getRevenueReport(params);
          break;
        case 'appointments':
          data = await this.getAppointmentReport(params);
          break;
        case 'doctors':
          data = await this.getDoctorPerformance(params);
          break;
        case 'stock':
          data = await this.getStockReport();
          break;
        default:
          throw new Error('Invalid report type');
      }

      if (format === 'csv') {
        return this.convertToCSV(data);
      }

      return JSON.stringify(data, null, 2);
    } catch (error) {
      console.error('Error in exportReport:', error);
      throw error;
    }
  }

  // Helper methods
  groupByPeriod(data, period) {
    const grouped = {};

    data.forEach(item => {
      let key;
      const date = new Date(item.created_at);

      switch (period) {
        case 'day':
          key = date.toISOString().split('T')[0];
          break;
        case 'week': {
            const week = this.getWeekNumber(date);
            key = `${date.getFullYear()}-W${week}`;
            break;
            }
        case 'month':
          key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
          break;
        default:
          key = date.toISOString().split('T')[0];
      }

      if (!grouped[key]) {
        grouped[key] = { period: key, total: 0, count: 0 };
      }

      grouped[key].total += item.total;
      grouped[key].count += 1;
    });

    return Object.values(grouped);
  }

  getWeekNumber(date) {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    const dayNum = d.getUTCDay() || 7;
    d.setUTCDate(d.getUTCDate() + 4 - dayNum);
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    return Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
  }

  convertToCSV(data) {
    // Simple CSV conversion - can be enhanced based on data structure
    const rows = [];
    
    if (Array.isArray(data)) {
      if (data.length === 0) return '';
      
      const headers = Object.keys(data[0]);
      rows.push(headers.join(','));
      
      data.forEach(item => {
        const values = headers.map(header => {
          const value = item[header];
          return typeof value === 'string' ? `"${value}"` : value;
        });
        rows.push(values.join(','));
      });
    }

    return rows.join('\n');
  }
}

module.exports = new ReportsService();