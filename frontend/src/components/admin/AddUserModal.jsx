import React, { useState } from 'react';
import { X } from 'lucide-react';
import { createUser } from '../../services/userService';

export default function AddUserModal({ onClose, onSuccess }) {
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    password: '',
    phone: '',
    role_id: '4', // default to Patient (matches role ids used elsewhere)
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    // basic validation to match backend requirements
    if (!formData.full_name.trim() || !formData.email.trim() || !formData.password.trim() || !formData.phone.trim() || !formData.role_id) {
      setError('Vui lòng điền tên, email, mật khẩu, số điện thoại và chọn vai trò');
      return;
    }

    try {
      setLoading(true);
        // ensure role_id is a number
        const payload = { ...formData, role_id: parseInt(formData.role_id, 10) };
        await createUser(payload);
      onSuccess();
      onClose();
    } catch (err) {
      setError(err.response?.data?.message || 'Tạo người dùng thất bại');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 max-h-96 overflow-y-auto">
        <div className="sticky top-0 flex items-center justify-between p-6 border-b border-gray-200 bg-white">
          <h2 className="text-xl font-bold text-gray-900">Thêm người dùng mới</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 transition"><X size={24} /></button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && <div className="text-sm text-red-600">{error}</div>}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Họ và tên</label>
            <input name="full_name" value={formData.full_name} onChange={handleChange} className="w-full px-3 py-2 border rounded" />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input name="email" type="email" value={formData.email} onChange={handleChange} className="w-full px-3 py-2 border rounded" />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Mật khẩu</label>
            <input name="password" type="password" value={formData.password} onChange={handleChange} className="w-full px-3 py-2 border rounded" />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Số điện thoại</label>
            <input name="phone" value={formData.phone} onChange={handleChange} className="w-full px-3 py-2 border rounded" />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Vai trò</label>
            <select name="role_id" value={formData.role_id} onChange={handleChange} className="w-full px-3 py-2 border rounded">
              <option value="">Chọn vai trò</option>
              <option value="1">ADMIN</option>
              <option value="2">DOCTOR</option>
              <option value="3">RECEPTIONIST</option>
              <option value="4">PHARMACIST</option>
              <option value="5">LAB_TECHNICIAN</option>
              <option value="6">PATIENT</option>
            </select>
          </div>

          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg">Hủy</button>
            <button type="submit" disabled={loading} className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg">{loading ? 'Đang tạo...' : 'Tạo'}</button>
          </div>
        </form>
      </div>
    </div>
  );
}
