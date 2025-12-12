// frontend/src/components/patient/AppointmentList.jsx
import React from 'react';
import { Calendar, Clock, User, MapPin, Phone, Mail, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { formatDate, formatTime } from '../../utils/helpers';
import Badge from '../common/Badge';

const getStatusColor = (status) => {
  const colors = {
    PENDING: 'bg-yellow-100 text-yellow-800',
    CONFIRMED: 'bg-blue-100 text-blue-800',
    CHECKED_IN: 'bg-purple-100 text-purple-800',
    IN_PROGRESS: 'bg-indigo-100 text-indigo-800',
    COMPLETED: 'bg-green-100 text-green-800',
    CANCELLED: 'bg-red-100 text-red-800',
    NO_SHOW: 'bg-gray-100 text-gray-800',
  };
  return colors[status] || 'bg-gray-100 text-gray-800';
};

const getStatusLabel = (status) => {
  const labels = {
    PENDING: 'Chờ xác nhận',
    CONFIRMED: 'Đã xác nhận',
    CHECKED_IN: 'Đã check-in',
    IN_PROGRESS: 'Đang khám',
    COMPLETED: 'Hoàn thành',
    CANCELLED: 'Đã hủy',
    NO_SHOW: 'Không đến',
  };
  return labels[status] || status;
};

const getStatusIcon = (status) => {
  switch (status) {
    case 'COMPLETED':
      return <CheckCircle className="w-5 h-5 text-green-600" />;
    case 'CANCELLED':
    case 'NO_SHOW':
      return <XCircle className="w-5 h-5 text-red-600" />;
    case 'PENDING':
      return <AlertCircle className="w-5 h-5 text-yellow-600" />;
    default:
      return <Clock className="w-5 h-5 text-blue-600" />;
  }
};

export default function AppointmentList({ 
  appointments = [], 
  onViewDetail, 
  onCancel,
  loading = false 
}) {
  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="bg-white rounded-lg shadow p-6 animate-pulse">
            <div className="h-6 bg-gray-200 rounded w-1/4 mb-4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          </div>
        ))}
      </div>
    );
  }

  if (appointments.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-12 text-center">
        <Calendar className="w-16 h-16 mx-auto text-gray-300 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          Chưa có lịch hẹn
        </h3>
        <p className="text-gray-500">
          Bạn chưa có lịch hẹn nào. Hãy đặt lịch khám ngay!
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {appointments.map((appointment) => (
        <div
          key={appointment.id}
          className="bg-white rounded-lg shadow hover:shadow-md transition-shadow"
        >
          <div className="p-6">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                {getStatusIcon(appointment.status)}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">
                    {appointment.doctor?.user?.full_name ? 
                      `BS. ${appointment.doctor.user.full_name}` : 
                      'Chưa có thông tin bác sĩ'}
                  </h3>
                  {appointment.doctor?.specialty && (
                    <p className="text-sm text-gray-600">
                      {appointment.doctor.specialty}
                    </p>
                  )}
                </div>
              </div>
              <span className={`px-3 py-1 text-xs font-medium rounded-full ${getStatusColor(appointment.status)}`}>
                {getStatusLabel(appointment.status)}
              </span>
            </div>

            {/* Details */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="flex items-center gap-2 text-gray-600">
                <Calendar className="w-4 h-4" />
                <span className="text-sm">
                  {formatDate(appointment.appointment_date)}
                </span>
              </div>
              <div className="flex items-center gap-2 text-gray-600">
                <Clock className="w-4 h-4" />
                <span className="text-sm">
                  {appointment.appointment_time}
                </span>
              </div>
              {appointment.room && (
                <div className="flex items-center gap-2 text-gray-600">
                  <MapPin className="w-4 h-4" />
                  <span className="text-sm">
                    {appointment.room.name || `Phòng ${appointment.room.room_number}`}
                  </span>
                </div>
              )}
              {appointment.doctor?.user?.phone && (
                <div className="flex items-center gap-2 text-gray-600">
                  <Phone className="w-4 h-4" />
                  <span className="text-sm">
                    {appointment.doctor.user.phone}
                  </span>
                </div>
              )}
            </div>

            {/* Reason */}
            {appointment.reason && (
              <div className="bg-gray-50 rounded-lg p-3 mb-4">
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Lý do khám:</span> {appointment.reason}
                </p>
              </div>
            )}

            {/* Notes */}
            {appointment.notes && (
              <div className="bg-blue-50 rounded-lg p-3 mb-4">
                <p className="text-sm text-blue-700">
                  <span className="font-medium">Ghi chú:</span> {appointment.notes}
                </p>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2 pt-4 border-t">
              {onViewDetail && (
                <button
                  onClick={() => onViewDetail(appointment)}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                >
                  Xem chi tiết
                </button>
              )}
              {onCancel && ['PENDING', 'CONFIRMED'].includes(appointment.status) && (
                <button
                  onClick={() => onCancel(appointment)}
                  className="px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors text-sm font-medium"
                >
                  Hủy lịch
                </button>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}