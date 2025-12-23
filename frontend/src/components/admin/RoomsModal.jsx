/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react';
import { Edit2, Trash2, Search, X, Home } from 'lucide-react';
import { roomService } from '../../services/roomService';

export default function RoomsModal({ isOpen, onClose }) {
  const [rooms, setRooms] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [addingRoom, setAddingRoom] = useState(false);
  const [editingRoom, setEditingRoom] = useState(null);
  const [deletingRoom, setDeletingRoom] = useState(null);
  const [detailRoom, setDetailRoom] = useState(null);
  const [formData, setFormData] = useState({ name: '', type: '', description: '', capacity: 1 });

  useEffect(() => {
    if (isOpen) {
      loadRooms();
    }
  }, [isOpen]);

  const loadRooms = async () => {
    setLoading(true);
    try {
      const res = await roomService.getAllRooms();
      const roomsData = Array.isArray(res) ? res : (res?.data?.data || res?.data || []);
      setRooms(Array.isArray(roomsData) ? roomsData : []);
    } catch (error) {
      console.error('Failed to load rooms:', error);
      setRooms([]);
    } finally {
      setLoading(false);
    }
  };

  const filteredRooms = rooms.filter(r =>
    r.name?.toLowerCase().includes(search.toLowerCase()) ||
    r.type?.toLowerCase().includes(search.toLowerCase())
  );

  const handleSave = async (e) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      alert('Tên phòng không được để trống');
      return;
    }

    setLoading(true);
    try {
      if (editingRoom) {
        await roomService.updateRoom(editingRoom.id, formData);
        alert('Cập nhật phòng thành công');
      } else {
        await roomService.createRoom(formData);
        alert('Thêm phòng thành công');
      }
      setAddingRoom(false);
      setEditingRoom(null);
      setFormData({ name: '', type: '', description: '', capacity: 1 });
      loadRooms();
    } catch (error) {
      alert(`Lỗi: ${error.response?.data?.message || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    setLoading(true);
    try {
      await roomService.deleteRoom(id);
      alert('Xóa phòng thành công');
      loadRooms();
      setDeletingRoom(null);
    } catch (error) {
      alert(`Lỗi: ${error.response?.data?.message || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50">
      {/* Background Overlay */}
      <div 
        className="absolute inset-0 bg-black bg-opacity-30"
        onClick={onClose}
      />
      
      {/* Sliding Panel from Right */}
      <div className="absolute right-0 top-0 h-full w-full md:w-2xl bg-white shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b shrink-0">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Quản lý phòng</h1>
            <p className="text-gray-600 mt-1">Quản lý danh sách phòng khám của phòng khám</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-900 transition"
            disabled={loading}
          >
            <X size={28} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-4 top-3 text-gray-400" size={20} />
              <input
                type="text"
                placeholder="Tìm kiếm theo tên hoặc loại..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
              />
            </div>

            {/* Add Button */}
            <div className="flex justify-end">
              <button
                onClick={() => {
                  setEditingRoom(null);
                  setFormData({ name: '', type: '', description: '', capacity: 1 });
                  setAddingRoom(true);
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
                disabled={loading}
              >
                + Thêm phòng
              </button>
            </div>

            {/* Table */}
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700 w-12">#</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Tên Phòng</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Loại</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Sức Chứa</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Mô Tả</th>
                      <th className="px-6 py-4 text-center text-sm font-semibold text-gray-700 w-24">Hành động</th>
                    </tr>
                  </thead>
                  <tbody>
                    {loading ? (
                      <tr>
                        <td colSpan="6" className="px-6 py-12 text-center">
                          <div className="flex justify-center">
                            <div className="spinner"></div>
                          </div>
                        </td>
                      </tr>
                    ) : filteredRooms.length === 0 ? (
                      <tr>
                        <td colSpan="6" className="px-6 py-12 text-center text-gray-500">
                          Không tìm thấy phòng nào
                        </td>
                      </tr>
                    ) : (
                      filteredRooms.map((r, i) => (
                        <tr key={r.id} className="border-b border-gray-200 hover:bg-gray-50 transition cursor-pointer" onClick={() => setDetailRoom(r)}>
                          <td className="px-6 py-4 text-sm text-gray-600">{i + 1}</td>
                          <td className="px-6 py-4 text-sm font-medium text-gray-900">{r.name}</td>
                          <td className="px-6 py-4 text-sm text-gray-600">{r.type || '-'}</td>
                          <td className="px-6 py-4 text-sm text-gray-600">{r.capacity || '-'}</td>
                          <td className="px-6 py-4 text-sm text-gray-600 truncate max-w-xs">{r.description || '-'}</td>
                          <td className="px-6 py-4 text-center" onClick={(e) => e.stopPropagation()}>
                            <div className="flex items-center justify-center gap-2">
                              <button
                                onClick={() => {
                                  setEditingRoom(r);
                                  setFormData({ name: r.name, type: r.type || '', description: r.description || '', capacity: r.capacity || 1 });
                                  setAddingRoom(true);
                                }}
                                className="p-2 text-blue-600 hover:bg-blue-100 rounded-lg transition"
                                title="Chỉnh sửa"
                              >
                                <Edit2 size={18} />
                              </button>
                              <button
                                onClick={() => setDeletingRoom(r)}
                                className="p-2 text-red-600 hover:bg-red-100 rounded-lg transition"
                                title="Xóa"
                              >
                                <Trash2 size={18} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Footer Info */}
            {filteredRooms.length > 0 && (
              <div className="text-sm text-gray-600">
                Hiển thị <span className="font-semibold">{filteredRooms.length}</span> phòng
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Add/Edit Modal */}
      {addingRoom && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="flex justify-between items-center p-6 border-b">
              <h2 className="text-xl font-bold">{editingRoom ? 'Sửa Phòng' : 'Thêm Phòng Mới'}</h2>
              <button
                onClick={() => {
                  setAddingRoom(false);
                  setEditingRoom(null);
                }}
                className="text-gray-500 hover:text-gray-900"
                disabled={loading}
              >
                <X size={24} />
              </button>
            </div>

            <form onSubmit={handleSave} className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">Tên Phòng *</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="VD: Phòng khám A1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={loading}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">Loại Phòng</label>
                <input
                  type="text"
                  value={formData.type}
                  onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                  placeholder="VD: Khám bệnh, Phòng chờ"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={loading}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">Sức Chứa</label>
                <input
                  type="number"
                  value={formData.capacity}
                  onChange={(e) => setFormData({ ...formData, capacity: parseInt(e.target.value) || 1 })}
                  placeholder="1"
                  min="1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={loading}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">Mô Tả</label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="Mô tả phòng..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows="2"
                  disabled={loading}
                />
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => {
                    setAddingRoom(false);
                    setEditingRoom(null);
                  }}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-900 hover:bg-gray-50 transition"
                  disabled={loading}
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
                  disabled={loading}
                >
                  {loading ? 'Đang lưu...' : editingRoom ? 'Cập nhật' : 'Thêm'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete Modal */}
      {deletingRoom && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="flex justify-between items-center p-6 border-b">
              <h2 className="text-xl font-bold">Xác nhận xóa</h2>
              <button
                onClick={() => setDeletingRoom(null)}
                className="text-gray-500 hover:text-gray-900"
                disabled={loading}
              >
                <X size={24} />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <p className="font-semibold text-gray-900">Xóa phòng "{deletingRoom.name}"?</p>
              <p className="text-sm text-gray-600">Hành động này không thể được hoàn tác.</p>

              <div className="flex gap-3 pt-4">
                <button
                  onClick={() => setDeletingRoom(null)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-900 hover:bg-gray-50 transition"
                  disabled={loading}
                >
                  Hủy
                </button>
                <button
                  onClick={() => handleDelete(deletingRoom.id)}
                  className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition disabled:opacity-50"
                  disabled={loading}
                >
                  {loading ? 'Đang xóa...' : 'Xóa'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Detail Modal */}
      {detailRoom && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="flex justify-between items-center p-6 border-b">
              <h2 className="text-xl font-bold">Chi tiết phòng</h2>
              <button
                onClick={() => setDetailRoom(null)}
                className="text-gray-500 hover:text-gray-900"
              >
                <X size={24} />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <p className="text-sm text-gray-600">Tên Phòng</p>
                <p className="font-semibold text-gray-900">{detailRoom.name}</p>
              </div>

              {detailRoom.type && (
                <div>
                  <p className="text-sm text-gray-600">Loại Phòng</p>
                  <p className="font-semibold text-gray-900">{detailRoom.type}</p>
                </div>
              )}

              {detailRoom.capacity && (
                <div>
                  <p className="text-sm text-gray-600">Sức Chứa</p>
                  <p className="font-semibold text-gray-900">{detailRoom.capacity}</p>
                </div>
              )}

              {detailRoom.description && (
                <div>
                  <p className="text-sm text-gray-600">Mô Tả</p>
                  <p className="text-gray-900">{detailRoom.description}</p>
                </div>
              )}

              <button
                onClick={() => setDetailRoom(null)}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition mt-4"
              >
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
