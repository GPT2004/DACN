/* eslint-disable no-unused-vars */
/* eslint-disable no-console */
import React, { useEffect, useState } from 'react';
import { Calendar, Plus, Search, Filter, Trash2, Edit2, Eye, CheckCircle, X } from 'lucide-react';
import { appointmentService } from '../../services/appointmentService';
import { patientService } from '../../services/patientService';
import AppointmentDetailModal from '../../components/receptionist/AppointmentDetailModal';

export default function AppointmentManagementPage() {
  const [appointments, setAppointments] = useState([]);
  const [filteredAppointments, setFilteredAppointments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [selectedAppointment, setSelectedAppointment] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [formData, setFormData] = useState({
    patient_id: '',
    doctor_id: '',
    appointment_date: '',
    appointment_time: '',
    notes: ''
  });
  const [patients, setPatients] = useState([]);
  const [doctors, setDoctors] = useState([]);
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('desc'); // 'asc' or 'desc'

  // Fetch initial data
  useEffect(() => {
    fetchAppointments();
    fetchPatients();
    fetchDoctors();
  }, []);

  // Filter appointments
  useEffect(() => {
    let filtered = appointments;

    if (searchTerm) {
      filtered = filtered.filter(apt =>
        (apt.patient?.user?.full_name || apt.patient?.full_name || '')
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        apt.doctor?.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        apt.id?.toString().includes(searchTerm)
      );
    }

    if (statusFilter) {
      filtered = filtered.filter(apt => apt.status === statusFilter);
    }

    // Apply sorting
    const orderFactor = sortOrder === 'asc' ? 1 : -1;
    const sorted = filtered.slice().sort((a, b) => {
      if (sortBy === 'created_at') {
        return (new Date(a.created_at) - new Date(b.created_at)) * orderFactor;
      }

      if (sortBy === 'patient') {
        const aName = (a.patient?.user?.full_name || a.patient?.full_name || '').toLowerCase();
        const bName = (b.patient?.user?.full_name || b.patient?.full_name || '').toLowerCase();
        return aName.localeCompare(bName) * orderFactor;
      }

      if (sortBy === 'status') {
        const aStatus = (a.status || '').toLowerCase();
        const bStatus = (b.status || '').toLowerCase();
        return aStatus.localeCompare(bStatus) * orderFactor;
      }

      return 0;
    });

    setFilteredAppointments(sorted);
  }, [appointments, searchTerm, statusFilter, sortBy, sortOrder]);

  const fetchAppointments = async () => {
    try {
      setLoading(true);
      const res = await appointmentService.getAppointments({ limit: 100, page: 1 });
      const data = Array.isArray(res?.data) ? res.data : res?.data?.appointments || [];
      // Sort appointments by newest first (created_at desc)
      const sorted = data.slice().sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      setAppointments(sorted);
    } catch (error) {
      console.error('Error fetching appointments:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPatients = async () => {
    try {
      const res = await patientService.getAllPatients({ limit: 100 });
      const data = res?.data?.patients || res?.data || [];
      setPatients(data);
    } catch (error) {
      console.error('Error fetching patients:', error);
    }
  };

  const fetchDoctors = async () => {
    try {
      // Mock doctors - thay thế bằng API thực tế
      setDoctors([
        { id: 1, user: { full_name: 'TS. Trần Văn A' } },
        { id: 2, user: { full_name: 'TS. Nguyễn Văn B' } },
        { id: 3, user: { full_name: 'TS. Phạm Văn C' } }
      ]);
    } catch (error) {
      console.error('Error fetching doctors:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editingId) {
        await appointmentService.updateAppointment(editingId, formData);
      } else {
        await appointmentService.createAppointment(formData);
      }
      fetchAppointments();
      resetForm();
    } catch (error) {
      console.error('Error saving appointment:', error);
      alert('Lỗi khi lưu lịch hẹn');
    }
  };

  const handleEdit = (appointment) => {
    setFormData({
      patient_id: appointment.patient_id,
      doctor_id: appointment.doctor_id,
      appointment_date: appointment.appointment_date?.split('T')[0],
      appointment_time: appointment.appointment_time || '',
      notes: appointment.notes || ''
    });
    setEditingId(appointment.id);
    setShowForm(true);
  };

  const handleViewDetail = (appointment) => {
    console.log('🔍 Xem chi tiết lịch hẹn:', appointment);
    setSelectedAppointment(appointment);
    setShowDetailModal(true);
  };

  const handleDelete = async (id, e) => {
    if (e) e.stopPropagation();
    console.log('🗑️ Xóa lịch hẹn ID:', id);
    if (window.confirm('Bạn có chắc muốn xóa lịch hẹn này?')) {
      try {
        await appointmentService.cancelAppointment(id);
        console.log('✅ Xóa thành công');
        fetchAppointments();
      } catch (error) {
        console.error('❌ Lỗi khi xóa lịch hẹn:', error);
        alert('Lỗi khi xóa lịch hẹn: ' + (error.response?.data?.message || error.message));
      }
    }
  };

  const handleCheckIn = async (id) => {
    try {
      await appointmentService.checkInAppointment(id);
      fetchAppointments();
    } catch (error) {
      console.error('Error checking in:', error);
      alert('Lỗi khi check-in');
    }
  };

  const resetForm = () => {
    setFormData({
      patient_id: '',
      doctor_id: '',
      appointment_date: '',
      appointment_time: '',
      notes: ''
    });
    setEditingId(null);
    setShowForm(false);
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
            <Calendar className="w-8 h-8 text-blue-600" />
            Quản Lý Lịch Hẹn
          </h1>
          <p className="text-gray-600 mt-1">Quản lý lịch hẹn khám của bệnh nhân</p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          Tạo Lịch Hẹn
        </button>
      </div>

      {/* Search & Filter */}
      <div className="bg-white rounded-lg shadow p-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="relative">
            <Search className="w-5 h-5 absolute left-3 top-3 text-gray-400" />
            <input
              type="text"
              placeholder="Tìm theo tên bệnh nhân, bác sĩ..."
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-gray-600" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Tất cả trạng thái</option>
              <option value="PENDING">Chờ xác nhận</option>
              <option value="CONFIRMED">Đã xác nhận</option>
              <option value="CHECKED_IN">Đã check-in</option>
              <option value="IN_PROGRESS">Đang khám</option>
              <option value="COMPLETED">Hoàn thành</option>
              <option value="CANCELLED">Đã hủy</option>
            </select>
            <div className="ml-4 flex items-center gap-2">
              <span className="text-sm text-gray-600">Sắp xếp:</span>
              <button
                type="button"
                onClick={() => {
                  if (sortBy === 'created_at') setSortOrder(prev => (prev === 'desc' ? 'asc' : 'desc'));
                  else {
                    setSortBy('created_at');
                    setSortOrder('desc');
                  }
                }}
                className={`px-3 py-1 rounded text-sm ${sortBy === 'created_at' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'}`}
              >
                Thời gian {sortBy === 'created_at' ? (sortOrder === 'desc' ? '▼' : '▲') : ''}
              </button>

              <button
                type="button"
                onClick={() => {
                  if (sortBy === 'patient') setSortOrder(prev => (prev === 'desc' ? 'asc' : 'desc'));
                  else {
                    setSortBy('patient');
                    setSortOrder('asc');
                  }
                }}
                className={`px-3 py-1 rounded text-sm ${sortBy === 'patient' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'}`}
              >
                Tên {sortBy === 'patient' ? (sortOrder === 'desc' ? '▼' : '▲') : ''}
              </button>

              <button
                type="button"
                onClick={() => {
                  if (sortBy === 'status') setSortOrder(prev => (prev === 'desc' ? 'asc' : 'desc'));
                  else {
                    setSortBy('status');
                    setSortOrder('asc');
                  }
                }}
                className={`px-3 py-1 rounded text-sm ${sortBy === 'status' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'}`}
              >
                Trạng thái {sortBy === 'status' ? (sortOrder === 'desc' ? '▼' : '▲') : ''}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Sort Toolbar (prominent) */}
      <div className="mt-3 mb-2 flex items-center justify-start gap-3">
        <span className="text-sm text-gray-600 mr-2">Sắp xếp nhanh:</span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              if (sortBy === 'created_at') setSortOrder(prev => (prev === 'desc' ? 'asc' : 'desc'));
              else {
                setSortBy('created_at');
                setSortOrder('desc');
              }
            }}
            className={`px-3 py-1 rounded border ${sortBy === 'created_at' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700'}`}
          >
            Thời gian {sortBy === 'created_at' ? (sortOrder === 'desc' ? '▼' : '▲') : ''}
          </button>

          <button
            onClick={() => {
              if (sortBy === 'patient') setSortOrder(prev => (prev === 'desc' ? 'asc' : 'desc'));
              else {
                setSortBy('patient');
                setSortOrder('asc');
              }
            }}
            className={`px-3 py-1 rounded border ${sortBy === 'patient' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700'}`}
          >
            Tên {sortBy === 'patient' ? (sortOrder === 'desc' ? '▼' : '▲') : ''}
          </button>

          <button
            onClick={() => {
              if (sortBy === 'status') setSortOrder(prev => (prev === 'desc' ? 'asc' : 'desc'));
              else {
                setSortBy('status');
                setSortOrder('asc');
              }
            }}
            className={`px-3 py-1 rounded border ${sortBy === 'status' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700'}`}
          >
            Trạng thái {sortBy === 'status' ? (sortOrder === 'desc' ? '▼' : '▲') : ''}
          </button>
        </div>
      </div>

      {/* Form Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg p-6 max-w-2xl w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-900">
                {editingId ? 'Sửa Lịch Hẹn' : 'Tạo Lịch Hẹn'}
              </h2>
              <button onClick={resetForm} className="text-gray-500 hover:text-gray-700">
                <X className="w-6 h-6" />
              </button>
            </div>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Bệnh nhân *</label>
                  <select
                    required
                    value={formData.patient_id}
                    onChange={(e) => setFormData({ ...formData, patient_id: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Chọn bệnh nhân</option>
                    {patients.map(p => (
                      <option key={p.id} value={p.id}>
                        {p.user?.full_name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Bác sĩ *</label>
                  <select
                    required
                    value={formData.doctor_id}
                    onChange={(e) => setFormData({ ...formData, doctor_id: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Chọn bác sĩ</option>
                    {doctors.map(d => (
                      <option key={d.id} value={d.id}>
                        {d.user?.full_name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Ngày khám *</label>
                  <input
                    type="date"
                    required
                    value={formData.appointment_date}
                    onChange={(e) => setFormData({ ...formData, appointment_date: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Giờ khám *</label>
                  <input
                    type="time"
                    required
                    value={formData.appointment_time}
                    onChange={(e) => setFormData({ ...formData, appointment_time: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Ghi chú</label>
                <textarea
                  value={formData.notes}
                  onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows="3"
                  placeholder="Ghi chú thêm về lịch hẹn..."
                />
              </div>
              <div className="flex justify-end gap-3 pt-4 border-t">
                <button
                  type="button"
                  onClick={resetForm}
                  className="px-4 py-2 border rounded-lg hover:bg-gray-50"
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  {editingId ? 'Cập Nhật' : 'Tạo'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Appointments Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-gray-500">Đang tải...</div>
        ) : filteredAppointments.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">ID</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Bệnh Nhân</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Bác Sĩ</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Ngày Giờ</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Trạng Thái</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Thao Tác</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {filteredAppointments.map((apt) => (
                  <tr 
                    key={apt.id} 
                    className="hover:bg-gray-50 cursor-pointer transition-colors"
                    onClick={() => handleViewDetail(apt)}
                  >
                    <td className="px-6 py-4 text-sm text-gray-900">#{apt.id}</td>
                    <td className="px-6 py-4 text-sm">
                      <div className="font-medium text-gray-900">{apt.patient?.user?.full_name}</div>
                      <div className="text-xs text-gray-500">{apt.patient?.user?.phone}</div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900">{apt.doctor?.user?.full_name}</td>
                    <td className="px-6 py-4 text-sm text-gray-900">
                      {new Date(apt.appointment_date).toLocaleDateString('vi-VN')}
                      {apt.appointment_time && ` - ${apt.appointment_time}`}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(apt.status)}`}>
                        {getStatusLabel(apt.status)}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <div className="flex items-center justify-center gap-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleViewDetail(apt);
                          }}
                          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors cursor-pointer text-sm font-medium"
                          title="Xem chi tiết"
                          type="button"
                        >
                          Sửa
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(apt.id, e);
                          }}
                          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors cursor-pointer text-sm font-medium"
                          title="Xóa lịch hẹn"
                          type="button"
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
        ) : (
          <div className="p-8 text-center text-gray-500">Không có lịch hẹn nào</div>
        )}
      </div>

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
