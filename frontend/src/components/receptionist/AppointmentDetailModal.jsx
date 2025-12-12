import React, { useState, useEffect } from 'react';
import { X, Trash2, CheckCircle, XCircle, RotateCw } from 'lucide-react';
import { appointmentService } from '../../services/appointmentService';

export default function AppointmentDetailModal({ appointment, onClose, onUpdate }) {
  const [loadingDetail, setLoadingDetail] = useState(true);
  const [appointmentData, setAppointmentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);

  const [lastUpdated, setLastUpdated] = useState(null);

  // Fetch appointment details (callable)
  const fetchDetails = async () => {
    if (!appointment?.id) return;
    try {
      setLoadingDetail(true);
      const response = await appointmentService.getAppointmentById(appointment.id);

      // Debug: log raw response to help troubleshoot why UI stays loading
      // eslint-disable-next-line no-console
      console.debug('[AppointmentDetailModal] raw response:', response);

      const serverPayload = response?.data ?? response;
      const appointmentObj = serverPayload?.data ?? serverPayload;

      // Debug: log normalized payload
      // eslint-disable-next-line no-console
      console.debug('[AppointmentDetailModal] normalized appointmentObj:', appointmentObj);

      setAppointmentData(appointmentObj || appointment);
      setLastUpdated(new Date());
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('[AppointmentDetailModal] fetchDetails error:', error);
      setAppointmentData(appointment);
    } finally {
      setLoadingDetail(false);
    }
  };

  // initial load
  useEffect(() => {
    fetchDetails();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appointment?.id]);

  if (!appointment) return null;

  const data = appointmentData || appointment;
  const patient = data.patient || {};
  const doctor = data.doctor || {};
  const timeslot = data.timeslot || {};

  // Normalize specialties: relation `doctorSpecialties` or array `specialties`
  const doctorSpecialtiesDisplay = (() => {
    if (Array.isArray(doctor.doctorSpecialties) && doctor.doctorSpecialties.length > 0) {
      return doctor.doctorSpecialties.map(ds => ds.specialty?.name).filter(Boolean).join(', ');
    }
    if (Array.isArray(doctor.specialties) && doctor.specialties.length > 0) {
      return doctor.specialties.join(', ');
    }
    return null;
  })();

  // Status helpers
  const getStatusColor = (status) => {
    const colors = {
      PENDING: 'bg-yellow-100 text-yellow-800',
      CONFIRMED: 'bg-blue-100 text-blue-800',
      CHECKED_IN: 'bg-purple-100 text-purple-800',
      IN_PROGRESS: 'bg-indigo-100 text-indigo-800',
      COMPLETED: 'bg-green-100 text-green-800',
      CANCELLED: 'bg-red-100 text-red-800',
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
    };
    return labels[status] || status;
  };

  // Action handlers
  const handleCheckIn = async () => {
    if (window.confirm('Xác nhận check-in cho lịch hẹn này?')) {
      setLoading(true);
      try {
        await appointmentService.checkInAppointment(data.id);
        onUpdate();
        onClose();
      } catch (error) {
        alert('Lỗi: ' + (error.response?.data?.message || error.message));
      } finally {
        setLoading(false);
      }
    }
  };

  const handleCancel = async () => {
    if (window.confirm('Hủy lịch hẹn này?')) {
      setLoading(true);
      try {
        await appointmentService.cancelAppointment(data.id, 'Hủy lịch hẹn');
        onUpdate();
        onClose();
      } catch (error) {
        alert('Lỗi: ' + (error.response?.data?.message || error.message));
      } finally {
        setLoading(false);
      }
    }
  };

  const handleDelete = async () => {
    setLoading(true);
    try {
      await appointmentService.deleteAppointment(data.id);
      onUpdate();
      onClose();
    } catch (error) {
      alert('Lỗi: ' + (error.response?.data?.message || error.message));
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (date) => {
    if (!date) return 'N/A';
    try {
      // if date is a combined ISO datetime, use it; otherwise parse date-only string
      const d = new Date(date);
      if (!isNaN(d.getTime())) return d.toLocaleDateString('vi-VN');
      return 'N/A';
    } catch {
      return 'N/A';
    }
  };

  const formatTime = (time) => {
    if (!time) return 'N/A';
    try {
      // If time is a simple string like "08:00" or "08:00:00", just extract HH:MM
      if (typeof time === 'string' && /^\d{2}:\d{2}(:\d{2})?$/.test(time)) {
        return time.substring(0, 5); // Return "HH:MM" format
      }
      
      // If time is an ISO datetime string, extract time portion
      if (typeof time === 'string' && /T/.test(time)) {
        // Try to extract HH:MM from ISO string
        const match = time.match(/T(\d{2}):(\d{2})/);
        if (match) {
          return `${match[1]}:${match[2]}`; // Return "HH:MM" format
        }
        
        // Fallback: parse as Date
        const t = new Date(time);
        if (!isNaN(t.getTime())) {
          const h = String(t.getHours()).padStart(2, '0');
          const m = String(t.getMinutes()).padStart(2, '0');
          return `${h}:${m}`;
        }
      }
      
      // If it's a Date object (from backend), try to extract time from ISO string
      if (time instanceof Date || (typeof time === 'string' && /^\d{4}-\d{2}-\d{2}/.test(time))) {
        const t = time instanceof Date ? time : new Date(time);
        if (!isNaN(t.getTime())) {
          const isoStr = t.toISOString();
          const timePart = isoStr.split('T')[1]; // Get "HH:MM:SS.xxxZ"
          return timePart.substring(0, 5); // Return "HH:MM"
        }
      }
      
      return 'N/A';
    } catch {
      return 'N/A';
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b p-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Chi tiết lịch hẹn</h2>
            <p className="text-sm text-gray-500">ID #{data.id}</p>
            {lastUpdated && (
              <p className="text-xs text-gray-400">Cập nhật: {new Date(lastUpdated).toLocaleTimeString('vi-VN')}</p>
            )}
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
            <X className="w-6 h-6" />
          </button>

          {/* Refresh button */}
          <button onClick={fetchDetails} className="ml-3 text-gray-500 hover:text-gray-700" title="Lấy dữ liệu mới">
            <RotateCw className="w-5 h-5" />
          </button>
        </div>

        {/* Loading notice (keeps showing while fetching) */}
        {loadingDetail && (
          <div className="p-6">
            <p className="text-center text-gray-600">Đang tải dữ liệu...</p>
          </div>
        )}

        {/* Content: always render immediately using passed-in `appointment` as fallback */}
        <div className="p-6 space-y-6">
            {/* Status */}
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <p className="text-sm text-gray-600">Trạng thái</p>
                <p className="font-semibold text-gray-900">{getStatusLabel(data.status)}</p>
              </div>
              <span className={`px-4 py-2 rounded-full text-sm font-medium ${getStatusColor(data.status)}`}>
                {getStatusLabel(data.status)}
              </span>
            </div>

            {/* Patient Info */}
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Thông tin bệnh nhân</h3>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Tên</p>
                    <p className="font-medium text-gray-900">{patient.full_name || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Điện thoại</p>
                    <p className="font-medium text-gray-900">{patient.phone || 'N/A'}</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Email</p>
                    <p className="font-medium text-gray-900">{patient.email || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Ngày sinh</p>
                    <p className="font-medium text-gray-900">{formatDate(patient.dob)}</p>
                  </div>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Địa chỉ</p>
                  <p className="font-medium text-gray-900">{patient.address || 'N/A'}</p>
                </div>
              </div>
            </div>

            {/* Doctor Info */}
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Thông tin bác sĩ</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-sm text-gray-600">Tên bác sĩ</p>
                  <p className="font-medium text-gray-900">{doctor.user?.full_name || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Chuyên khoa</p>
                  <p className="font-medium text-gray-900">{doctorSpecialtiesDisplay || 'N/A'}</p>
                </div>
              </div>
            </div>

            {/* Appointment Details */}
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Thông tin lịch hẹn</h3>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Ngày khám</p>
                    <p className="font-medium text-gray-900">{formatDate(data.appointment_date)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Giờ khám</p>
                    <p className="font-medium text-gray-900">{formatTime(data.appointment_time) || 'N/A'}</p>
                  </div>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Lý do khám</p>
                  <p className="font-medium text-gray-900">{data.reason || 'N/A'}</p>
                </div>
              </div>
            </div>

            {/* Timeslot Info */}
            {timeslot.id && (
              <div className="border-t pt-6 bg-blue-50 p-4 rounded-lg">
                <h3 className="text-sm font-semibold text-blue-900 mb-3">Chi tiết khung giờ</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-blue-700">Bắt đầu:</span>
                    <span className="font-medium text-blue-900">{formatTime(timeslot.start_time)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-700">Kết thúc:</span>
                    <span className="font-medium text-blue-900">{formatTime(timeslot.end_time)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-700">Sức chứa:</span>
                    <span className="font-medium text-blue-900">{timeslot.max_patients || 'N/A'}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Medical Records */}
            {patient.medical_records && patient.medical_records.length > 0 && (
              <div className="border-t pt-6 bg-green-50 p-4 rounded-lg">
                <h3 className="text-sm font-semibold text-green-900 mb-3">Hồ sơ bệnh nhân</h3>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {patient.medical_records.map((record, idx) => (
                    <div key={record.id} className="bg-white p-2 rounded border border-green-100 text-xs">
                      <p className="font-medium text-green-800">Hồ sơ #{idx + 1}</p>
                      {record.diagnosis && (
                        <p className="text-green-700">
                          <strong>Chẩn đoán:</strong> {record.diagnosis}
                        </p>
                      )}
                      {record.notes && (
                        <p className="text-green-700">
                          <strong>Ghi chú:</strong> {record.notes.substring(0, 50)}...
                        </p>
                      )}
                      <p className="text-gray-500 text-xs mt-1">{formatDate(record.created_at)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        
        

        {/* Footer Actions */}
        <div className="sticky bottom-0 bg-white border-t p-6 space-y-3">
          <div className="flex gap-3">
            {/* Check-in Button */}
            {(data.status === 'PENDING' || data.status === 'CONFIRMED') && (
              <button
                onClick={handleCheckIn}
                disabled={loading}
                className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-green-400 flex items-center justify-center gap-2"
              >
                <CheckCircle className="w-5 h-5" />
                Check-in
              </button>
            )}

            {/* Cancel Button */}
            {data.status !== 'CANCELLED' && data.status !== 'COMPLETED' && (
              <button
                onClick={handleCancel}
                disabled={loading}
                className="flex-1 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:bg-yellow-400 flex items-center justify-center gap-2"
              >
                <XCircle className="w-5 h-5" />
                Hủy
              </button>
            )}

            {/* Delete Button */}
            <button
              onClick={() => setShowConfirmDelete(true)}
              disabled={loading}
              className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-red-400 flex items-center justify-center gap-2"
            >
              <Trash2 className="w-5 h-5" />
              Xóa
            </button>
          </div>

          {/* Delete Confirmation */}
          {showConfirmDelete && (
            <div className="bg-red-50 p-4 rounded-lg border border-red-200">
              <p className="text-sm text-red-700 mb-3">Bạn có chắc muốn xóa lịch hẹn này?</p>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowConfirmDelete(false)}
                  disabled={loading}
                  className="flex-1 px-3 py-2 border rounded-lg text-gray-700 hover:bg-gray-50"
                >
                  Hủy
                </button>
                <button
                  onClick={handleDelete}
                  disabled={loading}
                  className="flex-1 px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-red-400"
                >
                  {loading ? 'Đang xóa...' : 'Xóa'}
                </button>
              </div>
            </div>
          )}

          {/* Close Button */}
          <button
            onClick={onClose}
            disabled={loading}
            className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
}
