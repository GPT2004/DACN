/* eslint-disable no-console */
/* eslint-disable react-hooks/exhaustive-deps */
import React, { useState, useEffect } from 'react';
import { Clock, Plus, Edit2, Trash2, X, Save } from 'lucide-react';

export default function TimeslotManagementPage() {
  const [timeslots, setTimeslots] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formData, setFormData] = useState({
    start_time: '',
    end_time: '',
    max_appointments: 5
  });
  const [loading, setLoading] = useState(false);

  // Mock data
  const mockTimeslots = [
    { id: 1, start_time: '08:00', end_time: '08:30', max_appointments: 5, available: 3 },
    { id: 2, start_time: '08:30', end_time: '09:00', max_appointments: 5, available: 2 },
    { id: 3, start_time: '09:00', end_time: '09:30', max_appointments: 5, available: 5 },
    { id: 4, start_time: '09:30', end_time: '10:00', max_appointments: 5, available: 1 },
    { id: 5, start_time: '10:00', end_time: '10:30', max_appointments: 5, available: 4 },
    { id: 6, start_time: '10:30', end_time: '11:00', max_appointments: 5, available: 3 },
    { id: 7, start_time: '13:00', end_time: '13:30', max_appointments: 5, available: 5 },
    { id: 8, start_time: '13:30', end_time: '14:00', max_appointments: 5, available: 2 },
    { id: 9, start_time: '14:00', end_time: '14:30', max_appointments: 5, available: 4 },
    { id: 10, start_time: '14:30', end_time: '15:00', max_appointments: 5, available: 3 },
    { id: 11, start_time: '15:00', end_time: '15:30', max_appointments: 5, available: 5 },
    { id: 12, start_time: '15:30', end_time: '16:00', max_appointments: 5, available: 1 },
  ];

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setTimeslots(mockTimeslots);
      setLoading(false);
    }, 500);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editingId) {
        // Update
        setTimeslots(timeslots.map(slot =>
          slot.id === editingId
            ? { ...slot, ...formData }
            : slot
        ));
      } else {
        // Create
        const newSlot = {
          id: Math.max(...timeslots.map(s => s.id), 0) + 1,
          ...formData,
          available: formData.max_appointments
        };
        setTimeslots([...timeslots, newSlot].sort((a, b) => a.start_time.localeCompare(b.start_time)));
      }
      resetForm();
    } catch (error) {
      console.error('Error saving timeslot:', error);
      alert('Lỗi khi lưu khung giờ');
    }
  };

  const handleEdit = (slot) => {
    setFormData({
      start_time: slot.start_time,
      end_time: slot.end_time,
      max_appointments: slot.max_appointments
    });
    setEditingId(slot.id);
    setShowForm(true);
  };

  const handleDelete = (id) => {
    if (window.confirm('Bạn có chắc muốn xóa khung giờ này?')) {
      setTimeslots(timeslots.filter(slot => slot.id !== id));
    }
  };

  const resetForm = () => {
    setFormData({
      start_time: '',
      end_time: '',
      max_appointments: 5
    });
    setEditingId(null);
    setShowForm(false);
  };

  const getUtilizationColor = (available, max) => {
    const usage = ((max - available) / max) * 100;
    if (usage === 0) return 'bg-green-100 text-green-800';
    if (usage < 50) return 'bg-blue-100 text-blue-800';
    if (usage < 80) return 'bg-orange-100 text-orange-800';
    return 'bg-red-100 text-red-800';
  };

  const getUtilizationLabel = (available, max) => {
    const usage = ((max - available) / max) * 100;
    return `${Math.round(usage)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
            <Clock className="w-8 h-8 text-blue-600" />
            Quản Lý Khung Giờ
          </h1>
          <p className="text-gray-600 mt-1">Cấu hình khung giờ khám bệnh</p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          Thêm Khung Giờ
        </button>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-gray-600 text-sm">Tổng khung giờ</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">{timeslots.length}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-gray-600 text-sm">Tổng chỗ trống</p>
          <p className="text-3xl font-bold text-green-600 mt-2">
            {timeslots.reduce((sum, slot) => sum + slot.available, 0)}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-gray-600 text-sm">Tổng khả năng</p>
          <p className="text-3xl font-bold text-blue-600 mt-2">
            {timeslots.reduce((sum, slot) => sum + slot.max_appointments, 0)}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-gray-600 text-sm">Đã đặt</p>
          <p className="text-3xl font-bold text-orange-600 mt-2">
            {timeslots.reduce((sum, slot) => sum + (slot.max_appointments - slot.available), 0)}
          </p>
        </div>
      </div>

      {/* Form Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-900">
                {editingId ? 'Sửa Khung Giờ' : 'Thêm Khung Giờ'}
              </h2>
              <button onClick={resetForm} className="text-gray-500 hover:text-gray-700">
                <X className="w-6 h-6" />
              </button>
            </div>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Giờ bắt đầu *</label>
                <input
                  type="time"
                  required
                  value={formData.start_time}
                  onChange={(e) => setFormData({ ...formData, start_time: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Giờ kết thúc *</label>
                <input
                  type="time"
                  required
                  value={formData.end_time}
                  onChange={(e) => setFormData({ ...formData, end_time: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Số lượng bệnh nhân tối đa *</label>
                <input
                  type="number"
                  required
                  min="1"
                  max="20"
                  value={formData.max_appointments}
                  onChange={(e) => setFormData({ ...formData, max_appointments: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  {editingId ? 'Cập Nhật' : 'Thêm'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Timeslots Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {loading ? (
          <div className="col-span-full p-8 text-center text-gray-500">Đang tải...</div>
        ) : timeslots.length > 0 ? (
          timeslots.map((slot) => (
            <div key={slot.id} className="bg-white rounded-lg shadow hover:shadow-lg transition p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Clock className="w-6 h-6 text-blue-600" />
                  <div>
                    <p className="text-lg font-bold text-gray-900">
                      {slot.start_time} - {slot.end_time}
                    </p>
                    <p className="text-xs text-gray-600">
                      {(() => {
                        const start = new Date(`2025-01-01 ${slot.start_time}`);
                        const end = new Date(`2025-01-01 ${slot.end_time}`);
                        const minutes = Math.round((end - start) / 60000);
                        return `${minutes} phút`;
                      })()}
                    </p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleEdit(slot)}
                    className="p-2 text-blue-600 hover:bg-blue-50 rounded"
                  >
                    <Edit2 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleDelete(slot.id)}
                    className="p-2 text-red-600 hover:bg-red-50 rounded"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Capacity Info */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Tổng chỗ:</span>
                  <span className="font-semibold text-gray-900">{slot.max_appointments}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Chỗ trống:</span>
                  <span className="font-semibold text-green-600">{slot.available}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Đã đặt:</span>
                  <span className="font-semibold text-orange-600">
                    {slot.max_appointments - slot.available}
                  </span>
                </div>

                {/* Utilization Bar */}
                <div className="mt-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-600">Mức sử dụng:</span>
                    <span className={`text-xs font-semibold px-2 py-1 rounded ${getUtilizationColor(slot.available, slot.max_appointments)}`}>
                      {getUtilizationLabel(slot.available, slot.max_appointments)}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{
                        width: `${((slot.max_appointments - slot.available) / slot.max_appointments) * 100}%`
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="col-span-full p-8 text-center text-gray-500">Không có khung giờ nào</div>
        )}
      </div>

      {/* Table View - For detailed view */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Khung Giờ</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Thời Lượng</th>
                <th className="px-6 py-3 text-right text-sm font-semibold text-gray-900">Tổng Chỗ</th>
                <th className="px-6 py-3 text-right text-sm font-semibold text-gray-900">Chỗ Trống</th>
                <th className="px-6 py-3 text-right text-sm font-semibold text-gray-900">Đã Đặt</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Mức Sử Dụng</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {timeslots.map((slot) => (
                <tr key={slot.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">
                    {slot.start_time} - {slot.end_time}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600">
                    {(() => {
                      const start = new Date(`2025-01-01 ${slot.start_time}`);
                      const end = new Date(`2025-01-01 ${slot.end_time}`);
                      const minutes = Math.round((end - start) / 60000);
                      return `${minutes} phút`;
                    })()}
                  </td>
                  <td className="px-6 py-4 text-sm text-right text-gray-900 font-semibold">
                    {slot.max_appointments}
                  </td>
                  <td className="px-6 py-4 text-sm text-right text-green-600 font-semibold">
                    {slot.available}
                  </td>
                  <td className="px-6 py-4 text-sm text-right text-orange-600 font-semibold">
                    {slot.max_appointments - slot.available}
                  </td>
                  <td className="px-6 py-4 text-sm">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${getUtilizationColor(slot.available, slot.max_appointments)}`}>
                      {getUtilizationLabel(slot.available, slot.max_appointments)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
