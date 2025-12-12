// frontend/src/components/admin/RoomManagement/RoomForm.jsx
import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { roomService } from '../../../services/roomService';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Modal from '../../common/Modal';

export default function RoomForm({ isOpen, onClose, room, onSuccess }) {
  const [formData, setFormData] = useState({
    room_number: '',
    name: '',
    type: '',
    floor: '',
    capacity: '',
    description: '',
    equipment: '',
    status: 'AVAILABLE',
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (room) {
      setFormData({
        room_number: room.room_number || '',
        name: room.name || '',
        type: room.type || '',
        floor: room.floor || '',
        capacity: room.capacity || '',
        description: room.description || '',
        equipment: room.equipment || '',
        status: room.status || 'AVAILABLE',
      });
    } else {
      resetForm();
    }
  }, [room]);

  const resetForm = () => {
    setFormData({
      room_number: '',
      name: '',
      type: '',
      floor: '',
      capacity: '',
      description: '',
      equipment: '',
      status: 'AVAILABLE',
    });
    setErrors({});
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.room_number.trim()) {
      newErrors.room_number = 'Vui lòng nhập mã phòng';
    }

    if (!formData.name.trim()) {
      newErrors.name = 'Vui lòng nhập tên phòng';
    }

    if (!formData.type) {
      newErrors.type = 'Vui lòng chọn loại phòng';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    try {
      setLoading(true);

      if (room) {
        await roomService.updateRoom(room.id, formData);
      } else {
        await roomService.createRoom(formData);
      }

      onSuccess();
      onClose();
      resetForm();
    } catch (error) {
      console.error('Submit error:', error);
      alert('Có lỗi xảy ra. Vui lòng thử lại!');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-800">
            {room ? 'Chỉnh sửa Phòng' : 'Thêm Phòng mới'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          <div className="grid grid-cols-2 gap-6">
            {/* Basic Info */}
            <div className="col-span-2">
              <h3 className="text-lg font-semibold mb-4">Thông tin cơ bản</h3>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Mã phòng <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.room_number}
                onChange={(e) => handleChange('room_number', e.target.value)}
                placeholder="VD: R101, E201"
              />
              {errors.room_number && (
                <p className="text-red-500 text-sm mt-1">{errors.room_number}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tên phòng <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value)}
                placeholder="Nhập tên phòng"
              />
              {errors.name && (
                <p className="text-red-500 text-sm mt-1">{errors.name}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Loại phòng <span className="text-red-500">*</span>
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                value={formData.type}
                onChange={(e) => handleChange('type', e.target.value)}
              >
                <option value="">Chọn loại phòng</option>
                <option value="CONSULTATION">Phòng khám</option>
                <option value="EXAMINATION">Phòng xét nghiệm</option>
                <option value="PROCEDURE">Phòng thủ thuật</option>
                <option value="EMERGENCY">Phòng cấp cứu</option>
                <option value="WAITING">Phòng chờ</option>
              </select>
              {errors.type && (
                <p className="text-red-500 text-sm mt-1">{errors.type}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Trạng thái
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                value={formData.status}
                onChange={(e) => handleChange('status', e.target.value)}
              >
                <option value="AVAILABLE">Sẵn sàng</option>
                <option value="OCCUPIED">Đang sử dụng</option>
                <option value="MAINTENANCE">Bảo trì</option>
                <option value="CLOSED">Đóng cửa</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tầng
              </label>
              <Input
                type="number"
                value={formData.floor}
                onChange={(e) => handleChange('floor', e.target.value)}
                placeholder="1, 2, 3..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Sức chứa
              </label>
              <Input
                type="number"
                value={formData.capacity}
                onChange={(e) => handleChange('capacity', e.target.value)}
                placeholder="Số người"
              />
            </div>

            {/* Details */}
            <div className="col-span-2 mt-4">
              <h3 className="text-lg font-semibold mb-4">Thông tin chi tiết</h3>
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Mô tả
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="3"
                value={formData.description}
                onChange={(e) => handleChange('description', e.target.value)}
                placeholder="Mô tả về phòng..."
              />
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Trang thiết bị
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="3"
                value={formData.equipment}
                onChange={(e) => handleChange('equipment', e.target.value)}
                placeholder="Liệt kê các trang thiết bị trong phòng..."
              />
            </div>
          </div>
        </form>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t bg-gray-50">
          <Button onClick={onClose} disabled={loading}>
            Hủy
          </Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Đang xử lý...' : room ? 'Cập nhật' : 'Thêm mới'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}