import React, { useEffect, useState } from 'react';
import PublicHeader from '../../components/common/PublicHeader';
import PatientPageHeader from '../../components/patient/PatientPageHeader';
import Footer from '../../components/patient/clinic/Footer';
import { appointmentService } from '../../services/appointmentService';
import { useAuth } from '../../context/AuthContext';
import { Calendar, Clock, User, FileText, RotateCw, ChevronDown } from 'lucide-react';
import AppointmentDetailModal from '../../components/patient/AppointmentDetailModal';

export default function AppointmentHistoryPage() {
  useAuth();
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedAppointment, setSelectedAppointment] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

  const fetchAppointments = async () => {
    try {
      setLoading(true);
      // Gọi API backend để lấy lịch sử từ DB
      const response = await appointmentService.getAppointmentHistory();
      const appts = response?.data?.appointments || response?.appointments || [];
      setAppointments(Array.isArray(appts) ? appts : []);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('Lỗi khi tải lịch sử lịch hẹn:', err);
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
      COMPLETED: 'bg-green-100 text-green-800 border border-green-300',
      CANCELLED: 'bg-red-100 text-red-800 border border-red-300',
      NO_SHOW: 'bg-orange-100 text-orange-800 border border-orange-300',
      PENDING: 'bg-yellow-100 text-yellow-800 border border-yellow-300',
      CONFIRMED: 'bg-blue-100 text-blue-800 border border-blue-300',
      CHECKED_IN: 'bg-purple-100 text-purple-800 border border-purple-300',
      IN_PROGRESS: 'bg-indigo-100 text-indigo-800 border border-indigo-300',
    };
    return colorMap[status] || 'bg-gray-100 text-gray-800';
  };

  const groupedByMonth = appointments.reduce((acc, apt) => {
    const date = new Date(apt.appointment_date);
    const key = date.toLocaleDateString('vi-VN', { year: 'numeric', month: 'long' });
    if (!acc[key]) acc[key] = [];
    acc[key].push(apt);
    return acc;
  }, {});

  const monthKeys = Object.keys(groupedByMonth).sort((a, b) => {
    const dateA = new Date(groupedByMonth[a][0].appointment_date);
    const dateB = new Date(groupedByMonth[b][0].appointment_date);
    return dateB - dateA;
  });

  return (
    <>
      <PublicHeader />
      <PatientPageHeader title="Lịch sử lịch hẹn" />

      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
        <div className="max-w-5xl mx-auto px-4 py-8">
          {/* Refresh Button */}
          <div className="mb-6 flex justify-end">
            <button
              onClick={fetchAppointments}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
            >
              <RotateCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Tải lại
            </button>
          </div>

          {loading ? (
            <div className="bg-white rounded-lg shadow p-8 text-center">
              <p className="text-gray-600">Đang tải lịch sử...</p>
            </div>
          ) : appointments.length === 0 ? (
            <div className="bg-white rounded-lg shadow p-12 text-center">
              <Calendar className="w-16 h-16 mx-auto text-gray-300 mb-4" />
              <h3 className="text-xl font-semibold text-gray-700 mb-2">Chưa có lịch hẹn trong quá khứ</h3>
              <p className="text-gray-500">Lịch sử lịch hẹn của bạn sẽ hiển thị ở đây sau khi bạn hoàn thành một cuộc hẹn.</p>
            </div>
          ) : (
            <div className="space-y-6">
              {monthKeys.map((monthKey) => (
                <div key={monthKey} className="bg-white rounded-lg shadow overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-500 to-indigo-600 px-6 py-3 text-white font-semibold">
                    {monthKey}
                  </div>
                  <div className="divide-y">
                    {groupedByMonth[monthKey].map((apt) => (
                      <div key={apt.id} className="p-6 hover:bg-gray-50 transition-colors">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            {/* Header: Date, Doctor, Status */}
                            <div className="flex flex-wrap items-center gap-4 mb-4">
                              <div className="flex items-center gap-2 text-gray-700">
                                <Calendar className="w-5 h-5 text-blue-500" />
                                <span className="font-semibold">{formatDate(apt.appointment_date)}</span>
                              </div>
                              <div className="flex items-center gap-2 text-gray-700">
                                <Clock className="w-5 h-5 text-purple-500" />
                                <span>{formatTime(apt.appointment_time)}</span>
                              </div>
                              <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(apt.status)}`}>
                                {getStatusLabel(apt.status)}
                              </span>
                            </div>

                            {/* Doctor Info */}
                            <div className="mb-4 pl-0 lg:pl-7">
                              <div className="flex items-center gap-2 mb-2">
                                <User className="w-4 h-4 text-gray-500" />
                                <span className="font-medium text-gray-800">
                                  {apt.doctor?.user?.full_name || 'N/A'}
                                </span>
                              </div>
                              {apt.doctor?.doctorSpecialties && apt.doctor.doctorSpecialties.length > 0 && (
                                <p className="text-sm text-gray-600 pl-6">
                                  Chuyên khoa: {apt.doctor.doctorSpecialties.map(ds => ds.specialty?.name).filter(Boolean).join(', ')}
                                </p>
                              )}
                            </div>

                            {/* Reason */}
                            {apt.reason && (
                              <div className="mb-4 pl-7 text-sm text-gray-600">
                                <strong>Lý do khám:</strong> {apt.reason}
                              </div>
                            )}

                            {/* Medical Records Summary */}
                            {apt.patient?.medical_records && apt.patient.medical_records.length > 0 && (
                              <div className="mb-4 pl-7 bg-green-50 p-3 rounded border border-green-200 text-sm">
                                <strong className="text-green-800">Hồ sơ khám:</strong>
                                {apt.patient.medical_records.map((record, idx) => (
                                  <div key={record.id} className="mt-2 text-green-700">
                                    <p className="font-medium">Hồ sơ #{idx + 1}</p>
                                    {record.diagnosis && <p>Chẩn đoán: {record.diagnosis}</p>}
                                    {record.notes && <p>Ghi chú: {record.notes.substring(0, 100)}...</p>}
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>

                          {/* Expand Button */}
                          <button
                            onClick={() => setExpandedId(expandedId === apt.id ? null : apt.id)}
                            className="ml-4 p-2 hover:bg-gray-200 rounded-lg transition-colors"
                          >
                            <ChevronDown
                              className={`w-5 h-5 text-gray-600 transition-transform ${
                                expandedId === apt.id ? 'rotate-180' : ''
                              }`}
                            />
                          </button>
                        </div>

                        {/* Expanded Details */}
                        {expandedId === apt.id && (
                          <div className="mt-4 pt-4 border-t border-gray-200 space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <p className="text-sm text-gray-600">Trạng thái chi tiết</p>
                                <p className="font-semibold text-gray-800">{getStatusLabel(apt.status)}</p>
                              </div>
                              <div>
                                <p className="text-sm text-gray-600">ID lịch hẹn</p>
                                <p className="font-semibold text-gray-800">#{apt.id}</p>
                              </div>
                            </div>

                            {apt.patient?.medical_records && apt.patient.medical_records.length > 0 && (
                              <div className="bg-blue-50 p-4 rounded border border-blue-200">
                                <h4 className="font-semibold text-blue-900 mb-3 flex items-center gap-2">
                                  <FileText className="w-4 h-4" />
                                  Chi tiết hồ sơ được chọn
                                </h4>
                                {apt.patient.medical_records.map((record, idx) => (
                                  <div key={record.id} className="mb-3 pb-3 border-b border-blue-200 last:border-0">
                                    <p className="font-medium text-blue-900">Hồ sơ #{idx + 1}</p>
                                    {record.diagnosis && (
                                      <p className="text-blue-800"><strong>Chẩn đoán:</strong> {record.diagnosis}</p>
                                    )}
                                    {record.notes && (
                                      <p className="text-blue-800"><strong>Ghi chú:</strong> {record.notes}</p>
                                    )}
                                    <p className="text-xs text-blue-600 mt-1">
                                      Ngày: {formatDate(record.created_at)}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            )}

                            <button
                              onClick={() => setSelectedAppointment(apt)}
                              className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
                            >
                              Xem chi tiết đầy đủ
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {selectedAppointment && (
        <AppointmentDetailModal
          appointment={selectedAppointment}
          onClose={() => setSelectedAppointment(null)}
          onUpdate={fetchAppointments}
        />
      )}

      <Footer />
    </>
  );
}
