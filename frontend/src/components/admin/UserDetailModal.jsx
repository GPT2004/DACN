import React from 'react';
import { X } from 'lucide-react';

export default function UserDetailModal({ user, onClose }) {
  if (!user) return null;

  const roleColors = {
    'ADMIN': 'bg-red-100 text-red-800',
    'DOCTOR': 'bg-blue-100 text-blue-800',
    'PATIENT': 'bg-green-100 text-green-800',
    'RECEPTIONIST': 'bg-purple-100 text-purple-800',
    'PHARMACIST': 'bg-orange-100 text-orange-800',
    'LAB_TECHNICIAN': 'bg-indigo-100 text-indigo-800',
  };

  const roleName = typeof user.role === 'object' ? user.role.name : user.role;
  const roleColor = roleColors[roleName] || 'bg-gray-100 text-gray-800';

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-screen overflow-y-auto">
        <div className="sticky top-0 flex items-center justify-between p-6 border-b border-gray-200 bg-white">
          <h2 className="text-xl font-bold text-gray-900">Chi tiết người dùng</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 transition">
            <X size={24} />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Thông tin cơ bản */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b">Thông tin cơ bản</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-gray-500">Tên người dùng</label>
                <p className="text-gray-900 font-medium">{user.full_name || user.fullName || user.name || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Email</label>
                <p className="text-gray-900">{user.email || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Vai trò</label>
                <p className="mt-1">
                  <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${roleColor}`}>
                    {roleName}
                  </span>
                </p>
              </div>
            </div>
          </div>

          {/* Thông tin bổ sung */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b">Thông tin bổ sung</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-gray-500">Số điện thoại</label>
                <p className="text-gray-900">{user.phone_number || user.phoneNumber || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Địa chỉ</label>
                <p className="text-gray-900">{user.address || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Ngày tạo</label>
                <p className="text-gray-900">{user.created_at ? new Date(user.created_at).toLocaleString('vi-VN') : 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Ngày cập nhật</label>
                <p className="text-gray-900">{user.updated_at ? new Date(user.updated_at).toLocaleString('vi-VN') : 'N/A'}</p>
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition font-medium"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
}
