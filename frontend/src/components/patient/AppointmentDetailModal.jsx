import React, { useState, useEffect } from 'react';
import { X, RotateCw } from 'lucide-react';
import { appointmentService } from '../../services/appointmentService';
import { useNavigate } from 'react-router-dom';

export default function AppointmentDetailModal({ appointment, onClose, onUpdate }) {
  const [loadingDetail, setLoadingDetail] = useState(true);
  const [appointmentData, setAppointmentData] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const navigate = useNavigate();

  const fetchDetails = async () => {
    if (!appointment?.id) return;
    try {
      setLoadingDetail(true);
      const response = await appointmentService.getAppointmentById(appointment.id);
      // eslint-disable-next-line no-console
      console.debug('[Patient AppointmentDetailModal] raw response:', response);

      const serverPayload = response?.data ?? response;
      const appointmentObj = serverPayload?.data ?? serverPayload;

      // eslint-disable-next-line no-console
      console.debug('[Patient AppointmentDetailModal] normalized appointmentObj:', appointmentObj);

      setAppointmentData(appointmentObj || appointment);
      setLastUpdated(new Date());
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('[Patient AppointmentDetailModal] fetchDetails error:', error);
      setAppointmentData(appointment);
    } finally {
      setLoadingDetail(false);
    }
  };

  useEffect(() => {
    fetchDetails();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appointment?.id]);

  if (!appointment) return null;

  const data = appointmentData || appointment;
  const patient = data.patient || {};
  const doctor = data.doctor || {};
  const timeslot = data.timeslot || {};

  const formatDate = (date) => {
    if (!date) return 'N/A';
    try {
      return new Date(date).toLocaleDateString('vi-VN');
    } catch {
      return 'N/A';
    }
  };

  const formatTime = (time) => {
    if (!time) return 'N/A';
    try {
      // String in HH:mm or HH:mm:ss
      if (typeof time === 'string') {
        const m = time.match(/^(\d{1,2}):(\d{2})(?::\d{2})?/);
        if (m) return `${m[1].padStart(2, '0')}:${m[2]}`;
        const d = new Date(time);
        if (!isNaN(d.getTime())) {
          const iso = d.toISOString();
          return `${iso.substring(11, 13)}:${iso.substring(14, 16)}`;
        }
        return time;
      }
      // Date object (Prisma TIME becomes Date anchored at 1970-01-01 UTC)
      if (time instanceof Date) {
        const iso = time.toISOString();
        return `${iso.substring(11, 13)}:${iso.substring(14, 16)}`;
      }
      // Numeric timestamp (ms or seconds)
      if (typeof time === 'number') {
        const ms = time < 1e12 ? time * 1000 : time;
        const d = new Date(ms);
        const iso = d.toISOString();
        return `${iso.substring(11, 13)}:${iso.substring(14, 16)}`;
      }
      // Plain object with hours/minutes
      if (typeof time === 'object' && time.hours !== undefined && time.minutes !== undefined) {
        return `${String(time.hours).padStart(2, '0')}:${String(time.minutes).padStart(2, '0')}`;
      }
      return String(time);
    } catch {
      return 'N/A';
    }
  };

  // Normalize specialties
  const doctorSpecialtiesDisplay = (() => {
    if (Array.isArray(doctor.doctorSpecialties) && doctor.doctorSpecialties.length > 0) {
      return doctor.doctorSpecialties.map(ds => ds.specialty?.name).filter(Boolean).join(', ');
    }
    if (Array.isArray(doctor.specialties) && doctor.specialties.length > 0) {
      return doctor.specialties.join(', ');
    }
    return null;
  })();

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white border-b p-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Chi tiết lịch hẹn</h2>
            <p className="text-sm text-gray-500">ID #{data.id}</p>
            {lastUpdated && (
              <p className="text-xs text-gray-400">Cập nhật: {new Date(lastUpdated).toLocaleTimeString('vi-VN')}</p>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button onClick={fetchDetails} className="text-gray-500 hover:text-gray-700" title="Lấy dữ liệu mới">
              <RotateCw className="w-5 h-5" />
            </button>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {loadingDetail && (
          <div className="p-6">
            <p className="text-center text-gray-600">Đang tải dữ liệu...</p>
          </div>
        )}

        <div className="p-6 space-y-6">
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <p className="text-sm text-gray-600">Trạng thái</p>
              <p className="font-semibold text-gray-900">{data.status || 'N/A'}</p>
            </div>
            <span className={`px-4 py-2 rounded-full text-sm font-medium bg-gray-100 text-gray-800`}>{data.status || 'N/A'}</span>
          </div>

          {/* Patient Info */}
          <div className="border-t pt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Thông tin bệnh nhân</h3>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Tên</p>
                  <p className="font-medium text-gray-900">{patient.full_name || patient.user?.full_name || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Điện thoại</p>
                  <p className="font-medium text-gray-900">{patient.phone || patient.user?.phone || 'N/A'}</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Email</p>
                  <p className="font-medium text-gray-900">{patient.email || patient.user?.email || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Ngày sinh</p>
                  <p className="font-medium text-gray-900">{formatDate(patient.dob || patient.user?.dob)}</p>
                </div>
              </div>
              <div>
                <p className="text-sm text-gray-600">Địa chỉ</p>
                <p className="font-medium text-gray-900">{patient.address || patient.new_street || 'N/A'}</p>
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

          {/* Medical Records (patient-facing) */}
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
                        <strong>Ghi chú:</strong> {record.notes.substring(0, 200)}
                      </p>
                    )}
                    <p className="text-gray-500 text-xs mt-1">{formatDate(record.created_at)}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="sticky bottom-0 bg-white border-t p-6">
          <div className="flex gap-3">
            <button
              onClick={() => {
                // Navigate to medical records page with patient and appointment prefilled
                const q = new URLSearchParams();
                if (patient.id) q.set('patient_id', String(patient.id));
                if (data.id) q.set('appointment_id', String(data.id));
                q.set('from_appointment', '1');
                // close modal then navigate
                onClose();
                navigate(`/doctor/medical-records?${q.toString()}`);
              }}
              className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
            >
              Thêm hồ sơ bệnh án
            </button>
            <button
              onClick={() => { onClose(); onUpdate && onUpdate(); }}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            >
              Đóng
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
