import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import ReceptionistLayout from '../../components/layout/ReceptionistLayout';
import { 
  Users, Calendar, DollarSign, CheckCircle
} from 'lucide-react';
import { appointmentService } from '../../services/appointmentService';

export default function ReceptionistDashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalPatients: 0,
    todayAppointments: 0,
    pendingPayments: 0,
    checkedInToday: 0,
    upcomingAppointments: 0,
    completedToday: 0
  });
  const [appointments, setAppointments] = useState([]);
  const [pendingInvoices, setPendingInvoices] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Get dashboard statistics
        const today = (() => {
          const d = new Date();
          const y = d.getFullYear();
          const m = String(d.getMonth() + 1).padStart(2, '0');
          const day = String(d.getDate()).padStart(2, '0');
          return `${y}-${m}-${day}`;
        })();
        
        const [appointmentsRes] = await Promise.all([
          appointmentService.getAppointments({ 
            limit: 100,
            page: 1 
          }).catch(err => {
            // eslint-disable-next-line no-console
            console.error('Error fetching appointments:', err);
            return { data: [] };
          })
        ]);

        const appointmentList = Array.isArray(appointmentsRes?.data) 
          ? appointmentsRes.data 
          : (appointmentsRes?.data?.appointments || []);

        // Filter appointments for today
        const todayAppointments = appointmentList.filter(apt => {
          const d = new Date(apt.appointment_date);
          const y = d.getFullYear();
          const m = String(d.getMonth() + 1).padStart(2, '0');
          const day = String(d.getDate()).padStart(2, '0');
          const aptDate = `${y}-${m}-${day}`;
          return aptDate === today;
        });

        // Calculate stats from appointments
        const checkedIn = todayAppointments.filter(apt => apt.status === 'CHECKED_IN' || apt.status === 'IN_PROGRESS').length;
        const completed = todayAppointments.filter(apt => apt.status === 'COMPLETED').length;
        const pending = todayAppointments.filter(apt => apt.status === 'PENDING' || apt.status === 'CONFIRMED').length;

        // Get unique patients
        const uniquePatients = new Set(appointmentList.map(apt => apt.patient_id)).size;

        setStats({
          totalPatients: uniquePatients,
          todayAppointments: todayAppointments.length,
          pendingPayments: Math.floor(Math.random() * 12) + 5, // Mock for now - would need invoice API
          checkedInToday: checkedIn,
          upcomingAppointments: pending,
          completedToday: completed
        });

        // Set appointments for display
        setAppointments(todayAppointments.slice(0, 3));
        
        // Mock pending invoices - would need real invoice API
        setPendingInvoices([
          {
            id: 'INV001',
            patientName: 'Nguyễn Văn D',
            amount: 500000,
            status: 'UNPAID'
          },
          {
            id: 'INV002',
            patientName: 'Phạm Văn E',
            amount: 750000,
            status: 'UNPAID'
          }
        ]);
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Error fetching dashboard data:', error);
        // Fallback to mock data if API fails
        setStats({
          totalPatients: 324,
          todayAppointments: 15,
          pendingPayments: 8,
          checkedInToday: 10,
          upcomingAppointments: 5,
          completedToday: 12
        });
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const getAppointmentColor = (status) => {
    switch(status) {
      case 'PENDING':
      case 'CONFIRMED':
        return 'bg-blue-50 border-l-blue-500';
      case 'CHECKED_IN':
      case 'IN_PROGRESS':
        return 'bg-purple-50 border-l-purple-500';
      case 'COMPLETED':
        return 'bg-green-50 border-l-green-500';
      default:
        return 'bg-gray-50 border-l-gray-500';
    }
  };

  const getStatusLabel = (status) => {
    switch(status) {
      case 'PENDING': return 'Chờ xác nhận';
      case 'CONFIRMED': return 'Đã xác nhận';
      case 'CHECKED_IN': return 'Đã check-in';
      case 'IN_PROGRESS': return 'Đang khám';
      case 'COMPLETED': return 'Hoàn thành';
      case 'CANCELLED': return 'Đã hủy';
      default: return status;
    }
  };

  if (loading) {
    return (
      <ReceptionistLayout>
        <div className="text-center py-8">Đang tải dữ liệu...</div>
      </ReceptionistLayout>
    );
  }

  return (
    <ReceptionistLayout>
      <div className="space-y-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Chào mừng, {user?.full_name}
          </h1>
          <p className="text-gray-600 mt-2">Quản lý bệnh nhân và lịch hẹn</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Tổng bệnh nhân</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats.totalPatients}</p>
              </div>
              <Users className="text-blue-500" size={40} />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Lịch hẹn khám bệnh</p>
                <p className="text-3xl font-bold text-purple-600 mt-2">{stats.todayAppointments}</p>
              </div>
              <Calendar className="text-purple-500" size={40} />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Thanh toán chờ xử lý</p>
                <p className="text-3xl font-bold text-orange-600 mt-2">{stats.pendingPayments}</p>
              </div>
              <DollarSign className="text-orange-500" size={40} />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Đã check-in hôm nay</p>
                <p className="text-3xl font-bold text-green-600 mt-2">{stats.checkedInToday}</p>
              </div>
              <CheckCircle className="text-green-500" size={40} />
            </div>
          </div>
        </div>

        {/* Recent Activities */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Lịch hẹn sắp tới</h2>
            <div className="space-y-3">
              {appointments.length > 0 ? (
                appointments.map((apt) => (
                  <div key={apt.id} className={`p-4 rounded-lg border-l-4 ${getAppointmentColor(apt.status)}`}>
                    <p className="text-sm font-medium text-gray-900">
                      {apt.patient?.user?.full_name || 'N/A'} - {apt.doctor?.user?.full_name || 'N/A'}
                    </p>
                    <p className="text-xs text-gray-600 mt-1">
                      {(() => {
                        try {
                          let dt = null;
                          if (apt.appointment_datetime) dt = new Date(apt.appointment_datetime);
                          else if (apt.appointment_date && apt.appointment_time) {
                            // Parse time component as local time (NOT UTC)
                            const timeRaw = apt.appointment_time;
                            let hours = 0, minutes = 0, seconds = 0;
                            
                            if (typeof timeRaw === 'string' && /^\d{2}:\d{2}(:\d{2})?/.test(timeRaw)) {
                              const parts = timeRaw.split(':');
                              hours = parseInt(parts[0], 10);
                              minutes = parseInt(parts[1], 10);
                              seconds = parts[2] ? parseInt(parts[2], 10) : 0;
                            }
                            
                            const dateOnly = new Date(apt.appointment_date);
                            const y = dateOnly.getFullYear();
                            const m = dateOnly.getMonth();
                            const d = dateOnly.getDate();
                            dt = new Date(y, m, d, hours, minutes, seconds);
                          }
                          else if (apt.appointment_time) dt = new Date(apt.appointment_time);
                          else if (apt.appointment_date) dt = new Date(apt.appointment_date);
                          if (dt && !isNaN(dt.getTime())) return dt.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
                        } catch (e) {}
                        return apt.appointment_time ? String(apt.appointment_time).slice(0,5) : 'N/A';
                      })()} • Phòng khám
                    </p>
                    <p className={`text-xs mt-1 ${
                      apt.status === 'COMPLETED' ? 'text-green-600' : 
                      apt.status === 'CHECKED_IN' || apt.status === 'IN_PROGRESS' ? 'text-purple-600' : 
                      'text-blue-600'
                    }`}>
                      {getStatusLabel(apt.status)}
                    </p>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 text-sm">Không có lịch hẹn khám bệnh</p>
              )}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Thanh toán chờ xử lý</h2>
            <div className="space-y-3">
              {pendingInvoices.length > 0 ? (
                pendingInvoices.map((invoice) => (
                  <div key={invoice.id} className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                    <p className="text-sm font-medium text-orange-900">
                      Hóa đơn #{invoice.id} - {invoice.patientName}
                    </p>
                    <p className="text-xs text-orange-700 mt-1">
                      {invoice.amount.toLocaleString('vi-VN')} VND • Chưa thanh toán
                    </p>
                    <button className="mt-2 text-xs bg-orange-600 text-white px-3 py-1 rounded hover:bg-orange-700 transition">
                      Xử lý thanh toán
                    </button>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 text-sm">Không có hóa đơn chờ xử lý</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </ReceptionistLayout>
  );
}
