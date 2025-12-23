/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
import React, { useEffect, useState } from 'react';
import DashboardLayout from '../../components/layout/DashboardLayout';
import RescheduleModal from '../../components/common/RescheduleModal';
import { getAppointmentsByPatient } from '../../services/appointmentService';
import { useAuth } from '../../context/AuthContext';

export default function AppointmentsPage() {
  const { user } = useAuth();
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [rescheduleAppointment, setRescheduleAppointment] = useState(null);

  useEffect(() => {
    const fetchAppointments = async () => {
      try {
        const result = await getAppointmentsByPatient();
        const appts = result?.data || result || [];
        setAppointments(Array.isArray(appts) ? appts : []);
      } catch (err) {
        console.error('Lỗi khi tải lịch hẹn:', err);
        setAppointments([]);
      } finally {
        setLoading(false);
      }
    };
    fetchAppointments();
  }, []);

  const formatDate = (date) => {
    return new Date(date).toLocaleDateString('vi-VN');
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
      PENDING: 'bg-yellow-50 border-yellow-200 text-yellow-800',
      CONFIRMED: 'bg-blue-50 border-blue-200 text-blue-800',
      CHECKED_IN: 'bg-green-50 border-green-200 text-green-800',
      IN_PROGRESS: 'bg-purple-50 border-purple-200 text-purple-800',
      COMPLETED: 'bg-gray-50 border-gray-200 text-gray-800',
      CANCELLED: 'bg-red-50 border-red-200 text-red-800',
      NO_SHOW: 'bg-orange-50 border-orange-200 text-orange-800'
    };
    return colorMap[status] || 'bg-gray-50 border-gray-200';
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="text-center py-8">Đang tải...</div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Lịch hẹn của tôi</h1>

        {appointments.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-6 text-center text-gray-500">
            Không có lịch hẹn nào
          </div>
        ) : (
          <div className="space-y-4">
            {appointments.map((appointment) => (
              <div
                key={appointment.id}
                className={`bg-white rounded-lg shadow p-4 border-l-4 ${
                  appointment.status === 'COMPLETED'
                    ? 'border-l-green-500'
                    : appointment.status === 'CANCELLED'
                    ? 'border-l-red-500'
                    : 'border-l-blue-500'
                }`}
              >
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Bác sĩ</p>
                    <p className="font-semibold">{appointment.doctor?.user?.full_name}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Ngày giờ</p>
                    <p className="font-semibold">
                      {formatDate(appointment.appointment_date)} {formatTime(appointment.appointment_time)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Lý do</p>
                    <p className="font-semibold">{appointment.reason || '-'}</p>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between">
                  <span className={`px-3 py-1 rounded text-sm font-medium border ${getStatusColor(appointment.status)}`}>
                    {getStatusLabel(appointment.status)}
                  </span>
                  {['PENDING', 'CONFIRMED'].includes(appointment.status) && (
                    <button
                      onClick={() => setRescheduleAppointment(appointment)}
                      className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
                    >
                      Dời lịch
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {rescheduleAppointment && (
          <RescheduleModal
            appointment={rescheduleAppointment}
            onSuccess={(updated) => {
              setAppointments(
                appointments.map((a) =>
                  a.id === updated.id ? updated : a
                )
              );
              setRescheduleAppointment(null);
              alert('Đã dời lịch thành công!');
            }}
            onCancel={() => setRescheduleAppointment(null)}
          />
        )}
      </div>
    </DashboardLayout>
  );
}
