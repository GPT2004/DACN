import React, { useEffect, useState } from 'react';
import AdminLayout from '../../components/admin/AdminLayout';
import { roomService } from '../../services/roomService';
import { Plus, Edit2, Trash2, Search, X } from 'lucide-react';

export default function RoomsPage() {
  const [rooms, setRooms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formData, setFormData] = useState({ name: '', type: '', description: '', capacity: 1 });

  useEffect(() => {
    loadRooms();
  }, []);

  const loadRooms = async () => {
    try {
      setLoading(true);
      // fetch many rooms so admin can scroll through the full list
      const res = await roomService.getAllRooms({ limit: 10000 });
      // Normalize different response shapes:
      // - service may return payload (i.e. { success, message, data: { rooms: [...] } })
      // - or payload.data (i.e. { rooms: [...] })
      // - or directly an array
      const roomsData = Array.isArray(res)
        ? res
        : (res?.rooms || res?.data?.rooms || res?.data?.data?.rooms || []);
      setRooms(Array.isArray(roomsData) ? roomsData : []);
    } catch (e) {
      console.error('Failed to load rooms', e);
      alert('Lỗi tải phòng: ' + (e?.message || 'Lỗi server'));
      setRooms([]);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenModal = (room = null) => {
    if (room) {
      setEditingId(room.id);
      setFormData({ name: room.name, type: room.type || '', description: room.description || '', capacity: room.capacity || 1 });
    } else {
      setEditingId(null);
      setFormData({ name: '', type: '', description: '', capacity: 1 });
    }
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setEditingId(null);
    setFormData({ name: '', type: '', description: '', capacity: 1 });
  };

  const handleSave = async () => {
    if (!formData.name.trim()) {
      alert('Tên phòng không được trống');
      return;
    }

    try {
      if (editingId) {
        await roomService.updateRoom(editingId, formData);
        alert('Cập nhật phòng thành công');
      } else {
        await roomService.createRoom(formData);
        alert('Thêm phòng thành công');
      }
      handleCloseModal();
      loadRooms();
    } catch (e) {
      alert('Lỗi: ' + (e?.message || 'Lỗi server'));
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc muốn xóa phòng này?')) return;

    try {
      await roomService.deleteRoom(id);
      alert('Xóa phòng thành công');
      loadRooms();
    } catch (e) {
      alert('Lỗi xóa: ' + (e?.message || 'Lỗi server'));
    }
  };

  const filtered = rooms.filter(r => 
    r.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    r.type?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <AdminLayout>
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Quản lý phòng khám</h1>
          <button 
            onClick={() => handleOpenModal()}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            <Plus size={20} /> Thêm phòng
          </button>
        </div>

        <div className="mb-4 flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-3 text-gray-400" size={18} />
            <input
              type="text"
              placeholder="Tìm kiếm phòng..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {loading ? (
          <div className="text-center py-8">Đang tải...</div>
        ) : filtered.length ? (
          <div className="overflow-x-auto">
            <div className="max-h-[60vh] overflow-y-auto">
              <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">ID</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Tên phòng</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Loại</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Sức chứa</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Mô tả</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Hành động</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(room => (
                  <tr key={room.id} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm text-gray-600">#{room.id}</td>
                    <td className="py-3 px-4 text-sm font-medium text-gray-900">{room.name}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">{room.type || '-'}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">{room.capacity || '-'}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">{room.description || '-'}</td>
                    <td className="py-3 px-4 text-sm space-x-2">
                      <button
                        onClick={() => handleOpenModal(room)}
                        className="text-blue-600 hover:text-blue-800 inline-flex items-center gap-1"
                        title="Sửa"
                      >
                        <Edit2 size={16} />
                      </button>
                      <button
                        onClick={() => handleDelete(room.id)}
                        className="text-red-600 hover:text-red-800 inline-flex items-center gap-1"
                        title="Xóa"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">Không có phòng nào</div>
        )}

        {/* Modal */}
        {showModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full">
              <div className="flex justify-between items-center p-6 border-b">
                <h2 className="text-xl font-bold">{editingId ? 'Sửa phòng' : 'Thêm phòng'}</h2>
                <button onClick={handleCloseModal} className="text-gray-500 hover:text-gray-700">
                  <X size={24} />
                </button>
              </div>

              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Tên phòng *</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="VD: Phòng khám A"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Loại phòng</label>
                  <input
                    type="text"
                    value={formData.type}
                    onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="VD: Khám bệnh, Phòng đợi"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Sức chứa</label>
                  <input
                    type="number"
                    value={formData.capacity}
                    onChange={(e) => setFormData({ ...formData, capacity: parseInt(e.target.value) || 1 })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Mô tả</label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Mô tả phòng"
                    rows="3"
                  />
                </div>
              </div>

              <div className="flex gap-3 p-6 border-t">
                <button
                  onClick={handleCloseModal}
                  className="flex-1 px-4 py-2 border rounded-lg text-gray-700 hover:bg-gray-50"
                >
                  Hủy
                </button>
                <button
                  onClick={handleSave}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  {editingId ? 'Cập nhật' : 'Thêm'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AdminLayout>
  );
}

