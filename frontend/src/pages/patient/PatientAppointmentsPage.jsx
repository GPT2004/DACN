import React, { useEffect, useState } from 'react';
import PublicHeader from '../../components/common/PublicHeader';
import PatientPageHeader from '../../components/patient/PatientPageHeader';
import Footer from '../../components/patient/clinic/Footer';
import appointmentService, { getAppointmentsByPatient } from '../../services/appointmentService';
import { useAuth } from '../../context/AuthContext';
import { Calendar, Clock, User, AlertCircle, RotateCw } from 'lucide-react';
import AppointmentDetailModal from '../../components/patient/AppointmentDetailModal';

export default function PatientAppointmentsPage() {
  useAuth();
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);

  const [selectedAppointment, setSelectedAppointment] = React.useState(null);

  const fetchAppointments = async () => {
    try {
      // If URL includes ?user_id=..., load that user's appointments (useful for patient support views)
      const params = new URLSearchParams(window.location.search);
      const userIdParam = params.get('user_id') || params.get('userId') || null;

      let result;
      if (userIdParam) {
        // Use general appointments endpoint filtered by patient_id (reception support view)
        result = await appointmentService.getAppointments({ patient_id: userIdParam, limit: 1000, page: 1 }).catch(() => ({ data: { appointments: [] } }));
        const appts = result?.data?.appointments || result?.data || result || [];
        setAppointments(Array.isArray(appts) ? appts : []);
      } else {
        // Use /appointments/me which now returns { upcoming, history, ... }
        result = await getAppointmentsByPatient({ page: 1, limit: 1000 }).catch(() => ({ data: { upcoming: [], history: [] } }));

        const data = result?.data || {};
        // If API returns upcoming/history, merge them so component logic stays the same
        if (Array.isArray(data.upcoming) || Array.isArray(data.history)) {
          const up = Array.isArray(data.upcoming) ? data.upcoming : [];
          const hs = Array.isArray(data.history) ? data.history : [];
          // Put upcoming first then history
          setAppointments([...up, ...hs]);
        } else {
          const appts = data?.appointments || data || [];
          setAppointments(Array.isArray(appts) ? appts : []);
        }
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('Lỗi khi tải lịch hẹn:', err);
      setAppointments([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAppointments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const formatDate = (date) => {
    return new Date(date).toLocaleDateString('vi-VN', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const formatTime = (timeVal) => {
    if (!timeVal) return '-';
    try {
      if (/^\d{2}:\d{2}/.test(timeVal)) {
        return timeVal.slice(0, 5);
      }
      if (typeof timeVal === 'string' && timeVal.includes('T')) {
        return new Date(timeVal).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
      }
      return String(timeVal).slice(0, 5);
    } catch {
      return String(timeVal).slice(0, 5);
    }
  };

  const getStatusLabel = (status) => {
    const statusMap = {
      PENDING: 'Chờ xác nhận',
      CONFIRMED: 'Đã xác nhận',
      CHECKED_IN: 'Đã check-in',
      IN_PROGRESS: 'Đang khám',
      COMPLETED: 'Hoàn thành',
      CANCELLED: 'Đã hủy',
      NO_SHOW: 'Vắng mặt'
    };
    return statusMap[status] || status;
  };

  const getStatusColor = (status) => {
    const colorMap = {
      PENDING: 'bg-yellow-100 text-yellow-800 border border-yellow-300',
      CONFIRMED: 'bg-blue-100 text-blue-800 border border-blue-300',
      CHECKED_IN: 'bg-green-100 text-green-800 border border-green-300',
      IN_PROGRESS: 'bg-purple-100 text-purple-800 border border-purple-300',
      COMPLETED: 'bg-gray-100 text-gray-800 border border-gray-300',
      CANCELLED: 'bg-red-100 text-red-800 border border-red-300',
      NO_SHOW: 'bg-orange-100 text-orange-800 border border-orange-300'
    };
    return colorMap[status] || 'bg-gray-100 text-gray-800';
  };

  // Treat CANCELLED / NO_SHOW / COMPLETED as history always
  const HISTORY_STATUSES = ['CANCELLED', 'NO_SHOW', 'COMPLETED'];

  const upcomingAppointments = appointments.filter(a => {
    // Build datetime from appointment_date + appointment_time
    let appointmentDateTime = new Date(a.appointment_date);
    if (a.appointment_time) {
      const [hours, minutes] = String(a.appointment_time).split(':');
      appointmentDateTime.setHours(parseInt(hours) || 0, parseInt(minutes) || 0, 0);
    }

    // Exclude any history statuses regardless of date
    if (HISTORY_STATUSES.includes(a.status)) return false;

    // Only upcoming if appointment datetime is strictly in future
    const now = new Date();
    const isUpcoming = appointmentDateTime > now;
    // eslint-disable-next-line no-console
    console.log(`[DEBUG] Appointment ${a.id}: date=${a.appointment_date} time=${a.appointment_time} status=${a.status} dateTime=${appointmentDateTime.toISOString()} now=${now.toISOString()} isUpcoming=${isUpcoming}`);
    return isUpcoming;
  });

  const pastAppointments = appointments.filter(a => {
    let appointmentDateTime = new Date(a.appointment_date);
    if (a.appointment_time) {
      const [hours, minutes] = String(a.appointment_time).split(':');
      appointmentDateTime.setHours(parseInt(hours) || 0, parseInt(minutes) || 0, 0);
    }

    // Any history-status appointment goes here, or any appointment in the past (<= now)
    const now = new Date();
    return HISTORY_STATUSES.includes(a.status) || appointmentDateTime <= now;
  });

  if (loading) {
    return (
      <div className="flex flex-col min-h-screen">
        <PublicHeader />
        <PatientPageHeader title="📅 Lịch khám của tôi" description="Quản lý và theo dõi các lịch hẹn khám chữa bệnh của bạn" />
        <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Đang tải lịch hẹn...</p>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen">
      <PublicHeader />
      <div className="max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between mt-6">
          <PatientPageHeader title="📅 Lịch khám của tôi" description="Quản lý và theo dõi các lịch hẹn khám chữa bệnh của bạn" />
          <div className="ml-4">
            <button
              onClick={fetchAppointments}
              className="inline-flex items-center gap-2 px-3 py-2 border rounded text-sm text-gray-700 hover:bg-gray-50"
            >
              <RotateCw className="w-4 h-4" />
              Tải lại
            </button>
          </div>
        </div>
      </div>
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">

        <div className="space-y-6">
          {/* Upcoming section: show message if no upcoming appointments */}
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">📆 Lịch sắp tới</h2>
            {upcomingAppointments.length === 0 ? (
              <div className="bg-white rounded-lg shadow-md p-8 text-center">
                <p className="text-gray-600 mb-4">Bạn chưa có lịch khám sắp tới.</p>
                <a
                  href="/patient/appointments/book"
                  className="inline-block bg-emerald-600 text-white px-6 py-2 rounded-lg hover:bg-emerald-700 transition"
                >
                  Đặt lịch hẹn ngay
                </a>
              </div>
            ) : (
              <div className="grid gap-4">
                {upcomingAppointments.map((appointment) => (
                  <div
                    key={appointment.id}
                    className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition border-l-4 border-l-emerald-500"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                      <div className="flex items-start gap-3">
                        <User className="w-5 h-5 text-emerald-600 mt-1 flex-shrink-0" />
                        <div>
                          <p className="text-sm text-gray-600 font-medium">Bác sĩ</p>
                          <p className="font-semibold text-gray-900">{appointment.doctor?.user?.full_name}</p>
                          {appointment.patient && (
                            <p className="text-sm text-gray-500 mt-1">Hồ sơ: {appointment.patient.full_name || appointment.patient.user?.full_name}</p>
                          )}
                          {appointment.doctor?.specialties && (
                            <p className="text-sm text-gray-500 mt-1">
                              {Array.isArray(appointment.doctor.specialties)
                                ? appointment.doctor.specialties.join(', ')
                                : appointment.doctor.specialties}
                            </p>
                          )}
                        </div>
                      </div>

                      <div className="flex items-start gap-3">
                        <Calendar className="w-5 h-5 text-emerald-600 mt-1 flex-shrink-0" />
                        <div>
                          <p className="text-sm text-gray-600 font-medium">Ngày khám</p>
                          <p className="font-semibold text-gray-900">{formatDate(appointment.appointment_date)}</p>
                        </div>
                      </div>

                      <div className="flex items-start gap-3">
                        <Clock className="w-5 h-5 text-emerald-600 mt-1 flex-shrink-0" />
                        <div>
                          <p className="text-sm text-gray-600 font-medium">Thời gian</p>
                          <p className="font-semibold text-gray-900">{formatTime(appointment.appointment_time)}</p>
                        </div>
                      </div>

                      <div>
                        <p className="text-sm text-gray-600 font-medium mb-2">Trạng thái</p>
                        <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(appointment.status)}`}>
                          {getStatusLabel(appointment.status)}
                        </span>
                      </div>
                    </div>

                    {appointment.reason && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <p className="text-sm text-gray-600">
                          <strong>Lý do khám:</strong> {appointment.reason}
                        </p>
                      </div>
                    )}
                    <div className="mt-4 flex justify-end">
                      <button
                        onClick={() => setSelectedAppointment(appointment)}
                        className="text-emerald-600 hover:underline text-sm"
                      >
                        Xem chi tiết
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* History section: show if there are past appointments */}
          {pastAppointments.length > 0 && (
            <section>
              <h2 className="text-2xl font-bold text-gray-900 mb-4">📜 Lịch sử khám</h2>
              <div className="grid gap-4">
                {pastAppointments.map((appointment) => (
                  <div
                    key={appointment.id}
                    className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition border-l-4 border-l-gray-400 opacity-75"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                      <div className="flex items-start gap-3">
                        <User className="w-5 h-5 text-gray-500 mt-1 flex-shrink-0" />
                        <div>
                          <p className="text-sm text-gray-600 font-medium">Bác sĩ</p>
                          <p className="font-semibold text-gray-900">{appointment.doctor?.user?.full_name}</p>
                          {appointment.doctor?.specialties && (
                            <p className="text-sm text-gray-500 mt-1">
                              {Array.isArray(appointment.doctor.specialties)
                                ? appointment.doctor.specialties.join(', ')
                                : appointment.doctor.specialties}
                            </p>
                          )}
                        </div>
                      </div>

                      <div className="flex items-start gap-3">
                        <Calendar className="w-5 h-5 text-gray-500 mt-1 flex-shrink-0" />
                        <div>
                          <p className="text-sm text-gray-600 font-medium">Ngày khám</p>
                          <p className="font-semibold text-gray-900">{formatDate(appointment.appointment_date)}</p>
                        </div>
                      </div>

                      <div className="flex items-start gap-3">
                        <Clock className="w-5 h-5 text-gray-500 mt-1 flex-shrink-0" />
                        <div>
                          <p className="text-sm text-gray-600 font-medium">Thời gian</p>
                          <p className="font-semibold text-gray-900">{formatTime(appointment.appointment_time)}</p>
                        </div>
                      </div>

                      <div>
                        <p className="text-sm text-gray-600 font-medium mb-2">Trạng thái</p>
                        <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(appointment.status)}`}>
                          {getStatusLabel(appointment.status)}
                        </span>
                      </div>
                    </div>

                    {appointment.reason && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <p className="text-sm text-gray-600">
                          <strong>Lý do khám:</strong> {appointment.reason}
                        </p>
                      </div>
                    )}
                    <div className="mt-4 flex justify-end">
                      <button
                        onClick={() => setSelectedAppointment(appointment)}
                        className="text-emerald-600 hover:underline text-sm"
                      >
                        Xem chi tiết
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </main>
      {selectedAppointment && (
        <AppointmentDetailModal
          appointment={selectedAppointment}
          onClose={() => setSelectedAppointment(null)}
          onUpdate={fetchAppointments}
        />
      )}
      <Footer />
    </div>
  );
}
