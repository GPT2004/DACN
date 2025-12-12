import React, { useState } from 'react';
import { X, CheckCircle2, Clock, User, Stethoscope } from 'lucide-react';

/**
 * CheckInModal - Xử lý check-in bệnh nhân
 */
export default function CheckInModal({ appointment, onClose, onCheckInSuccess }) {
  const [loading, setLoading] = useState(false);
  const [notes, setNotes] = useState('');

  const handleCheckIn = async () => {
    try {
      setLoading(true);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      onCheckInSuccess({
        appointmentId: appointment.id,
        checkInTime: new Date().toISOString(),
        notes,
        patientId: appointment.patient_id
      });
      
      onClose();
    } catch (error) {
      alert('Lỗi check-in: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full">
        <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-6 border-b flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
            <CheckCircle2 size={24} className="text-purple-600" /> Check-in bệnh nhân
          </h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
            <X size={24} />
          </button>
        </div>

        <div className="p-6 space-y-4">
          {/* Appointment Summary */}
          <div className="bg-purple-50 p-4 rounded-lg border-l-4 border-purple-500">
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <User size={18} className="text-purple-600" />
                <div>
                  <p className="text-xs text-gray-600">Bệnh nhân</p>
                  <p className="text-sm font-semibold text-gray-900">{appointment?.patient?.user?.full_name}</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Stethoscope size={18} className="text-purple-600" />
                <div>
                  <p className="text-xs text-gray-600">Bác sĩ</p>
                  <p className="text-sm font-semibold text-gray-900">BS. {appointment?.doctor?.user?.full_name}</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Clock size={18} className="text-purple-600" />
                <div>
                  <p className="text-xs text-gray-600">Thời gian lịch hẹn</p>
                  <p className="text-sm font-semibold text-gray-900">
                    {(() => {
                      try {
                        let dt = null;
                        if (appointment?.appointment_datetime) dt = new Date(appointment.appointment_datetime);
                        else if (appointment?.appointment_date && appointment?.appointment_time) {
                          // Parse time component as local time (NOT UTC)
                          const timeRaw = appointment.appointment_time;
                          let hours = 0, minutes = 0, seconds = 0;
                          
                          if (typeof timeRaw === 'string' && /^\d{2}:\d{2}(:\d{2})?/.test(timeRaw)) {
                            const parts = timeRaw.split(':');
                            hours = parseInt(parts[0], 10);
                            minutes = parseInt(parts[1], 10);
                            seconds = parts[2] ? parseInt(parts[2], 10) : 0;
                          }
                          
                          const dateOnly = new Date(appointment.appointment_date);
                          const y = dateOnly.getFullYear();
                          const m = dateOnly.getMonth();
                          const d = dateOnly.getDate();
                          dt = new Date(y, m, d, hours, minutes, seconds);
                        }
                        else if (appointment?.appointment_time) dt = new Date(appointment.appointment_time);
                        else if (appointment?.appointment_date) dt = new Date(appointment.appointment_date);
                        if (dt && !isNaN(dt.getTime())) return dt.toLocaleString('vi-VN', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
                      } catch (e) {}
                      return 'N/A';
                    })()}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Reason */}
          {appointment?.reason && (
            <div className="bg-gray-50 p-3 rounded-lg">
              <p className="text-xs text-gray-600 font-semibold mb-1">Lý do khám</p>
              <p className="text-sm text-gray-800">{appointment.reason}</p>
            </div>
          )}

          {/* Check-in Notes */}
          <div>
            <label className="text-sm font-semibold text-gray-900 mb-2 block">Ghi chú (tùy chọn)</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Ghi chú về tình trạng bệnh nhân khi check-in..."
              className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-600 resize-none text-sm"
              rows="3"
            />
          </div>

          {/* Confirmation */}
          <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
            <p className="text-sm text-yellow-800">
              <strong>⚠️ Lưu ý:</strong> Confirm check-in sẽ cập nhật trạng thái bệnh nhân thành "Đã check-in"
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4 border-t">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 font-medium"
            >
              Hủy
            </button>
            <button
              onClick={handleCheckIn}
              disabled={loading}
              className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 font-medium flex items-center justify-center gap-2"
            >
              <CheckCircle2 size={16} />
              {loading ? 'Đang xử lý...' : 'Xác nhận check-in'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
