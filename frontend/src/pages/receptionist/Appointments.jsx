/* eslint-disable no-console */
import React, { useEffect, useState } from 'react';
import { appointmentService } from '../../services/appointmentService';
import AppointmentDetailModal from '../../components/receptionist/AppointmentDetailModal';

export default function AppointmentsPage() {
  const [appts, setAppts] = useState([]);
  const [selectedAppointment, setSelectedAppointment] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAppointments();
  }, []);

  const fetchAppointments = async () => {
    try {
      setLoading(true);
      // Lấy tất cả lịch hẹn, không filter status
      const res = await appointmentService.getAppointments({ limit: 100, page: 1 });
      const data = Array.isArray(res?.data) ? res.data : res?.data?.appointments || [];
      setAppts(data);
    } catch (error) {
      console.error('Error fetching appointments:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetail = (appointment) => {
    console.log('✅ Click vào lịch hẹn:', appointment);
    setSelectedAppointment(appointment);
    setShowDetailModal(true);
  };

  const handleDelete = async (id) => {
    if (window.confirm('Bạn có chắc muốn xóa lịch hẹn này?')) {
      try {
        await appointmentService.cancelAppointment(id);
        fetchAppointments();
      } catch (error) {
        console.error('Error deleting appointment:', error);
        alert('Lỗi khi xóa lịch hẹn: ' + (error.response?.data?.message || error.message));
      }
    }
  };

  const getStatusLabel = (status) => {
    const labels = {
      PENDING: 'Chờ xác nhận',
      CONFIRMED: 'Đã xác nhận',
      CHECKED_IN: 'Đã check-in',
      IN_PROGRESS: 'Đang khám',
      COMPLETED: 'Hoàn thành',
      CANCELLED: 'Đã hủy'
    };
    return labels[status] || status;
  };

  const handleCheckIn = async (id) => {
    try {
      await appointmentService.checkInAppointment(id);
      alert('Check-in thành công! Bệnh nhân chuyển đến lịch hẹn của bác sĩ.');
      fetchAppointments();
    } catch (error) {
      console.error('Error checking in:', error);
      alert('Lỗi khi check-in: ' + (error.response?.data?.message || error.message));
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'PENDING': return 'bg-yellow-100 text-yellow-800';
      case 'CONFIRMED': return 'bg-blue-100 text-blue-800';
      case 'CHECKED_IN': return 'bg-purple-100 text-purple-800';
      case 'IN_PROGRESS': return 'bg-indigo-100 text-indigo-800';
      case 'COMPLETED': return 'bg-green-100 text-green-800';
      case 'CANCELLED': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className='p-6'>
      <h1 className='text-2xl font-bold mb-4'>Danh sách lịch hẹn</h1>
      {loading ? (
        <div className='text-center text-gray-500'>Đang tải...</div>
      ) : appts.length === 0 ? (
        <div className='text-center text-gray-500'>Không có lịch hẹn nào</div>
      ) : (
        <div className='bg-white rounded-lg shadow overflow-hidden'>
          <div className='overflow-x-auto'>
            <table className='w-full'>
              <thead className='bg-gray-50 border-b'>
                <tr>
                  <th className='px-6 py-3 text-left text-sm font-semibold text-gray-900'>ID</th>
                  <th className='px-6 py-3 text-left text-sm font-semibold text-gray-900'>Bệnh Nhân</th>
                  <th className='px-6 py-3 text-left text-sm font-semibold text-gray-900'>Bác Sĩ</th>
                  <th className='px-6 py-3 text-left text-sm font-semibold text-gray-900'>Ngày Giờ</th>
                  <th className='px-6 py-3 text-left text-sm font-semibold text-gray-900'>Trạng Thái</th>
                  <th className='px-6 py-3 text-left text-sm font-semibold text-gray-900'>Thao Tác</th>
                </tr>
              </thead>
              <tbody className='divide-y'>
                {appts.map((apt) => (
                  <tr 
                    key={apt.id} 
                    className='hover:bg-gray-50 cursor-pointer transition-colors'
                    onClick={() => handleViewDetail(apt)}
                  >
                    <td className='px-6 py-4 text-sm text-gray-900'>#{apt.id}</td>
                    <td className='px-6 py-4 text-sm'>
                      <div className='font-medium text-gray-900'>{apt.patient?.user?.full_name}</div>
                      <div className='text-xs text-gray-500'>{apt.patient?.user?.phone}</div>
                    </td>
                    <td className='px-6 py-4 text-sm text-gray-900'>{apt.doctor?.user?.full_name}</td>
                    <td className='px-6 py-4 text-sm text-gray-900'>
                      {new Date(apt.appointment_date).toLocaleDateString('vi-VN')}
                      {apt.appointment_time && ` - ${apt.appointment_time}`}
                    </td>
                    <td className='px-6 py-4 text-sm'>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(apt.status)}`}>
                        {getStatusLabel(apt.status)}
                      </span>
                    </td>
                    <td className='px-6 py-4 text-sm'>
                      <div className='flex items-center justify-center gap-2'>
                        {apt.status === 'PENDING' && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              e.preventDefault();
                              handleCheckIn(apt.id);
                            }}
                            className='px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors cursor-pointer text-sm font-medium'
                            title='Check-in bệnh nhân'
                            type='button'
                          >
                            Check-in
                          </button>
                        )}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            handleViewDetail(apt);
                          }}
                          className='px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors cursor-pointer text-sm font-medium'
                          title='Xem chi tiết và chỉnh sửa'
                          type='button'
                        >
                          Sửa
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            handleDelete(apt.id);
                          }}
                          className='px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors cursor-pointer text-sm font-medium'
                          title='Xóa lịch hẹn'
                          type='button'
                        >
                          Xóa
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Appointment Detail Modal */}
      {showDetailModal && selectedAppointment && (
        <AppointmentDetailModal
          appointment={selectedAppointment}
          onClose={() => {
            setShowDetailModal(false);
            setSelectedAppointment(null);
          }}
          onUpdate={() => {
            fetchAppointments();
            setShowDetailModal(false);
            setSelectedAppointment(null);
          }}
        />
      )}
    </div>
  );
}
