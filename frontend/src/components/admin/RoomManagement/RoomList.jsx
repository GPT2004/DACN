/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
// frontend/src/components/admin/RoomManagement/RoomList.jsx
import React, { useState, useEffect } from 'react';
import { Search, Plus, Edit, Trash2, MapPin, ToggleLeft, ToggleRight } from 'lucide-react';
import { roomService } from '../../../services/roomService';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Badge from '../../common/Badge';
import Loader from '../../common/Loader';
import Pagination from '../../common/Pagination';

export default function RoomList({ onEdit, onCreate }) {
  const [rooms, setRooms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [roomType, setRoomType] = useState('');

  useEffect(() => {
    fetchRooms();
  }, [page, searchTerm, roomType]);

  const fetchRooms = async () => {
    try {
      setLoading(true);
      const response = await roomService.getAllRooms({
        page,
        limit: 10,
        search: searchTerm,
        type: roomType
      });
      setRooms(response.data.rooms || []);
      setTotalPages(response.data.pagination?.totalPages || 1);
    } catch (error) {
      console.error('Fetch rooms error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleStatus = async (id, currentStatus) => {
    try {
      const newStatus = currentStatus === 'AVAILABLE' ? 'MAINTENANCE' : 'AVAILABLE';
      await roomService.updateRoomStatus(id, newStatus);
      fetchRooms();
    } catch (error) {
      console.error('Toggle status error:', error);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa phòng này?')) return;
    
    try {
      await roomService.deleteRoom(id);
      fetchRooms();
    } catch (error) {
      console.error('Delete room error:', error);
      alert('Không thể xóa phòng này!');
    }
  };

  const getStatusBadge = (status) => {
    const statusMap = {
      AVAILABLE: { text: 'Sẵn sàng', color: 'bg-green-100 text-green-800' },
      OCCUPIED: { text: 'Đang sử dụng', color: 'bg-blue-100 text-blue-800' },
      MAINTENANCE: { text: 'Bảo trì', color: 'bg-yellow-100 text-yellow-800' },
      CLOSED: { text: 'Đóng cửa', color: 'bg-red-100 text-red-800' },
    };
    
    const badge = statusMap[status] || statusMap.AVAILABLE;
    return <span className={`px-2 py-1 text-xs font-medium rounded-full ${badge.color}`}>{badge.text}</span>;
  };

  const getRoomTypeLabel = (type) => {
    const typeMap = {
      CONSULTATION: 'Phòng khám',
      EXAMINATION: 'Phòng xét nghiệm',
      PROCEDURE: 'Phòng thủ thuật',
      EMERGENCY: 'Phòng cấp cứu',
      WAITING: 'Phòng chờ',
    };
    return typeMap[type] || type;
  };

  if (loading) {
    return <Loader />;
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Quản lý Phòng khám</h2>
        <Button onClick={onCreate}>
          <Plus className="w-4 h-4 mr-2" />
          Thêm Phòng
        </Button>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="flex-1">
          <Input
            placeholder="Tìm kiếm theo tên phòng, mã phòng..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <select
          className="px-4 py-2 border border-gray-300 rounded-lg"
          value={roomType}
          onChange={(e) => setRoomType(e.target.value)}
        >
          <option value="">Tất cả loại phòng</option>
          <option value="CONSULTATION">Phòng khám</option>
          <option value="EXAMINATION">Phòng xét nghiệm</option>
          <option value="PROCEDURE">Phòng thủ thuật</option>
          <option value="EMERGENCY">Phòng cấp cứu</option>
          <option value="WAITING">Phòng chờ</option>
        </select>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-green-50 rounded-lg p-4">
          <div className="text-sm text-gray-600">Sẵn sàng</div>
          <div className="text-2xl font-bold text-green-600">
            {rooms.filter(r => r.status === 'AVAILABLE').length}
          </div>
        </div>
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="text-sm text-gray-600">Đang sử dụng</div>
          <div className="text-2xl font-bold text-blue-600">
            {rooms.filter(r => r.status === 'OCCUPIED').length}
          </div>
        </div>
        <div className="bg-yellow-50 rounded-lg p-4">
          <div className="text-sm text-gray-600">Bảo trì</div>
          <div className="text-2xl font-bold text-yellow-600">
            {rooms.filter(r => r.status === 'MAINTENANCE').length}
          </div>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="text-sm text-gray-600">Tổng phòng</div>
          <div className="text-2xl font-bold text-gray-800">
            {rooms.length}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Mã phòng
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Tên phòng
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Loại phòng
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Vị trí
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Sức chứa
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Trạng thái
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                Thao tác
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {rooms.map((room) => (
              <tr key={room.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">
                    {room.room_number}
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="text-sm font-medium text-gray-900">
                    {room.name}
                  </div>
                  {room.description && (
                    <div className="text-sm text-gray-500">
                      {room.description}
                    </div>
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <Badge text={getRoomTypeLabel(room.type)} />
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center text-sm text-gray-500">
                    <MapPin className="w-4 h-4 mr-1" />
                    {room.floor ? `Tầng ${room.floor}` : 'N/A'}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {room.capacity || 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getStatusBadge(room.status)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => onEdit(room)}
                      className="text-yellow-600 hover:text-yellow-900"
                      title="Chỉnh sửa"
                    >
                      <Edit className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => handleToggleStatus(room.id, room.status)}
                      className="text-green-600 hover:text-green-900"
                      title="Đổi trạng thái"
                    >
                      {room.status === 'AVAILABLE' ? (
                        <ToggleRight className="w-5 h-5" />
                      ) : (
                        <ToggleLeft className="w-5 h-5" />
                      )}
                    </button>
                    <button
                      onClick={() => handleDelete(room.id)}
                      className="text-red-600 hover:text-red-900"
                      title="Xóa"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {rooms.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">Không có dữ liệu</p>
          </div>
        )}
      </div>

      {/* Pagination */}
      <Pagination
        currentPage={page}
        totalPages={totalPages}
        onPageChange={setPage}
      />
    </div>
  );
}