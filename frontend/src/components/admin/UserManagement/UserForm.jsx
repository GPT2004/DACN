// frontend/src/components/admin/UserManagement/UserForm.jsx
import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { userService } from '../../../services/userService';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Modal from '../../common/Modal';

export default function UserForm({ isOpen, onClose, user, onSuccess }) {
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    phone: '',
    password: '',
    dob: '',
    gender: 'Male',
    address: '',
    role_id: '',
    is_active: true,
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [roles] = useState([
    { id: 1, name: 'Admin' },
    { id: 2, name: 'Doctor' },
    { id: 3, name: 'Patient' },
    { id: 4, name: 'Receptionist' },
    { id: 5, name: 'Pharmacist' },
    { id: 6, name: 'LabTech' },
  ]);

  useEffect(() => {
    if (user) {
      setFormData({
        full_name: user.full_name || '',
        email: user.email || '',
        phone: user.phone || '',
        password: '',
        dob: user.dob || '',
        gender: user.gender || 'Male',
        address: user.address || '',
        role_id: user.role?.id || '',
        is_active: user.is_active !== false,
      });
    } else {
      resetForm();
    }
  }, [user]);

  const resetForm = () => {
    setFormData({
      full_name: '',
      email: '',
      phone: '',
      password: '',
      dob: '',
      gender: 'Male',
      address: '',
      role_id: '',
      is_active: true,
    });
    setErrors({});
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.full_name.trim()) {
      newErrors.full_name = 'Vui lòng nhập họ tên';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Vui lòng nhập email';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email không hợp lệ';
    }

    if (!formData.phone.trim()) {
      newErrors.phone = 'Vui lòng nhập số điện thoại';
    }

    if (!user && !formData.password) {
      newErrors.password = 'Vui lòng nhập mật khẩu';
    } else if (!user && formData.password.length < 6) {
      newErrors.password = 'Mật khẩu phải có ít nhất 6 ký tự';
    }

    if (!formData.role_id) {
      newErrors.role_id = 'Vui lòng chọn vai trò';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    try {
      setLoading(true);

      if (user) {
        await userService.updateUser(user.id, formData);
      } else {
        await userService.createUser(formData);
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
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-800">
            {user ? 'Chỉnh sửa Người dùng' : 'Thêm Người dùng mới'}
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
                Họ tên <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.full_name}
                onChange={(e) => handleChange('full_name', e.target.value)}
                placeholder="Nhập họ tên"
              />
              {errors.full_name && (
                <p className="text-red-500 text-sm mt-1">{errors.full_name}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email <span className="text-red-500">*</span>
              </label>
              <Input
                type="email"
                value={formData.email}
                onChange={(e) => handleChange('email', e.target.value)}
                placeholder="email@example.com"
                disabled={!!user}
              />
              {errors.email && (
                <p className="text-red-500 text-sm mt-1">{errors.email}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Số điện thoại <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.phone}
                onChange={(e) => handleChange('phone', e.target.value)}
                placeholder="0123456789"
              />
              {errors.phone && (
                <p className="text-red-500 text-sm mt-1">{errors.phone}</p>
              )}
            </div>

            {!user && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Mật khẩu <span className="text-red-500">*</span>
                </label>
                <Input
                  type="password"
                  value={formData.password}
                  onChange={(e) => handleChange('password', e.target.value)}
                  placeholder="Nhập mật khẩu"
                />
                {errors.password && (
                  <p className="text-red-500 text-sm mt-1">{errors.password}</p>
                )}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Ngày sinh
              </label>
              <Input
                type="date"
                value={formData.dob}
                onChange={(e) => handleChange('dob', e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Giới tính
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                value={formData.gender}
                onChange={(e) => handleChange('gender', e.target.value)}
              >
                <option value="Male">Nam</option>
                <option value="Female">Nữ</option>
                <option value="Other">Khác</option>
              </select>
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Địa chỉ
              </label>
              <Input
                value={formData.address}
                onChange={(e) => handleChange('address', e.target.value)}
                placeholder="Nhập địa chỉ"
              />
            </div>

            {/* Role & Status */}
            <div className="col-span-2 mt-4">
              <h3 className="text-lg font-semibold mb-4">Vai trò & Trạng thái</h3>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Vai trò <span className="text-red-500">*</span>
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                value={formData.role_id}
                onChange={(e) => handleChange('role_id', e.target.value)}
              >
                <option value="">Chọn vai trò</option>
                {roles.map((role) => (
                  <option key={role.id} value={role.id}>
                    {role.name}
                  </option>
                ))}
              </select>
              {errors.role_id && (
                <p className="text-red-500 text-sm mt-1">{errors.role_id}</p>
              )}
            </div>

            <div className="flex items-center">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                  checked={formData.is_active}
                  onChange={(e) => handleChange('is_active', e.target.checked)}
                />
                <span className="ml-2 text-sm text-gray-700">
                  Tài khoản hoạt động
                </span>
              </label>
            </div>
          </div>
        </form>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t bg-gray-50">
          <Button onClick={onClose} disabled={loading}>
            Hủy
          </Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Đang xử lý...' : user ? 'Cập nhật' : 'Thêm mới'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}