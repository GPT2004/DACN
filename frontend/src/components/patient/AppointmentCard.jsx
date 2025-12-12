// frontend/src/components/patient/AppointmentCard.jsx
import React from 'react';
import { Calendar, Clock, User, MapPin, AlertCircle, CheckCircle } from 'lucide-react';
import { formatDate, formatTime } from '../../utils/helpers';
import Badge from '../common/Badge';
import Button from '../common/Button';

export default function AppointmentCard({ appointment, onCancel, onView }) {
  const getStatusColor = (status) => {
    const colors = {
      PENDING: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      CONFIRMED: 'bg-blue-100 text-blue-800 border-blue-200',
      CHECKED_IN: 'bg-purple-100 text-purple-800 border-purple-200',
      IN_PROGRESS: 'bg-indigo-100 text-indigo-800 border-indigo-200',
      COMPLETED: 'bg-green-100 text-green-800 border-green-200',
      CANCELLED: 'bg-red-100 text-red-800 border-red-200',
      NO_SHOW: 'bg-gray-100 text-gray-800 border-gray-200',
    };
    return colors[status] || 'bg-gray-100 text-gray-800 border-gray-200';
  };

  const getStatusText = (status) => {
    const texts = {
      PENDING: 'Chờ xác nhận',
      CONFIRMED: 'Đã xác nhận',
      CHECKED_IN: 'Đã check-in',
      IN_PROGRESS: 'Đang khám',
      COMPLETED: 'Hoàn thành',
      CANCELLED: 'Đã hủy',
      NO_SHOW: 'Không đến',
    };
    return texts[status] || status;
  };

  const canCancel = ['PENDING', 'CONFIRMED'].includes(appointment.status);
  const isUpcoming = ['PENDING', 'CONFIRMED'].includes(appointment.status);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
            isUpcoming ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'
          }`}>
            <Calendar className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">
              {formatDate(appointment.appointment_date)}
            </h3>
            <div className="flex items-center gap-2 text-sm text-gray-600 mt-1">
              <Clock className="w-4 h-4" />
              {appointment.appointment_time}
            </div>
          </div>
        </div>
        <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(appointment.status)}`}>
          {getStatusText(appointment.status)}
        </span>
      </div>

      {/* Doctor Info */}
      <div className="space-y-3 mb-4 pb-4 border-b border-gray-100">
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <User className="w-4 h-4 text-gray-400" />
          <span className="font-medium">Bác sĩ:</span>
          <span>{appointment.doctor?.user?.full_name || 'Chưa phân bổ'}</span>
        </div>
        {appointment.doctor?.specialty && (
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs">
              {appointment.doctor.specialty}
            </span>
          </div>
        )}
        {appointment.room && (
          <div className="flex items-center gap-2 text-sm text-gray-700">
            <MapPin className="w-4 h-4 text-gray-400" />
            <span>Phòng: {appointment.room.name}</span>
          </div>
        )}
      </div>

      {/* Reason */}
      {appointment.reason && (
        <div className="mb-4">
          <p className="text-sm text-gray-600 line-clamp-2">
            <span className="font-medium">Lý do khám:</span> {appointment.reason}
          </p>
        </div>
      )}

      {/* Status Message */}
      {appointment.status === 'PENDING' && (
        <div className="flex items-start gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg mb-4">
          <AlertCircle className="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-yellow-800">
            Lịch hẹn đang chờ xác nhận. Chúng tôi sẽ thông báo khi được xác nhận.
          </p>
        </div>
      )}

      {appointment.status === 'CONFIRMED' && (
        <div className="flex items-start gap-2 p-3 bg-green-50 border border-green-200 rounded-lg mb-4">
          <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-green-800">
            Lịch hẹn đã được xác nhận. Vui lòng đến đúng giờ.
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2">
        <Button
          onClick={() => onView(appointment)}
          className="flex-1"
          variant="outline"
        >
          Xem chi tiết
        </Button>
        {canCancel && (
          <Button
            onClick={() => onCancel(appointment.id)}
            variant="danger"
            className="flex-1"
          >
            Hủy lịch
          </Button>
        )}
      </div>

      {/* Notes */}
      {appointment.notes && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <p className="text-xs text-gray-500">
            <span className="font-medium">Ghi chú:</span> {appointment.notes}
          </p>
        </div>
      )}
    </div>
  );
}