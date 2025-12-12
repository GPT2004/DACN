// frontend/src/components/admin/UserManagement/UserDetail.jsx
import React from 'react';
import { X, User, Mail, Phone, Calendar, MapPin, Shield } from 'lucide-react';
import { formatDate } from '../../../utils/helpers';
import Modal from '../../common/Modal';
import Badge from '../../common/Badge';

export default function UserDetail({ isOpen, onClose, user }) {
  if (!user) return null;

  const getRoleBadgeColor = (role) => {
    const roleColors = {
      Admin: 'bg-purple-100 text-purple-800',
      Doctor: 'bg-blue-100 text-blue-800',
      Patient: 'bg-green-100 text-green-800',
      Receptionist: 'bg-yellow-100 text-yellow-800',
      Pharmacist: 'bg-pink-100 text-pink-800',
      LabTech: 'bg-indigo-100 text-indigo-800',
    };
    return roleColors[role] || 'bg-gray-100 text-gray-800';
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b bg-gradient-to-r from-blue-500 to-blue-600">
          <h2 className="text-2xl font-bold text-white">Thông tin Người dùng</h2>
          <button
            onClick={onClose}
            className="text-white hover:text-gray-200"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          {/* Profile Section */}
          <div className="flex items-start gap-6 mb-8">
            <img
              src={user.avatar_url || '/avatar-placeholder.png'}
              alt={user.full_name}
              className="w-32 h-32 rounded-full object-cover border-4 border-blue-100"
            />
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">
                {user.full_name}
              </h3>
              <div className="flex items-center gap-3 mb-4">
                <span className={`px-3 py-1 text-sm font-medium rounded-full ${getRoleBadgeColor(user.role?.name)}`}>
                  {user.role?.name}
                </span>
                <Badge text={user.is_active ? 'Hoạt động' : 'Ngừng hoạt động'} />
              </div>
              <div className="space-y-2">
                <div className="flex items-center text-gray-600">
                  <Mail className="w-5 h-5 mr-3" />
                  {user.email}
                </div>
                <div className="flex items-center text-gray-600">
                  <Phone className="w-5 h-5 mr-3" />
                  {user.phone}
                </div>
                {user.dob && (
                  <div className="flex items-center text-gray-600">
                    <Calendar className="w-5 h-5 mr-3" />
                    Ngày sinh: {formatDate(user.dob)}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Details */}
          <div className="space-y-6">
            {/* Personal Info */}
            <div>
              <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                <User className="w-5 h-5 mr-2 text-blue-600" />
                Thông tin cá nhân
              </h4>
              <div className="bg-gray-50 rounded-lg p-4 grid grid-cols-2 gap-4">
                <div>
                  <span className="text-gray-600">Giới tính:</span>
                  <span className="font-medium ml-2">
                    {user.gender === 'Male' ? 'Nam' : user.gender === 'Female' ? 'Nữ' : 'Khác'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Ngày sinh:</span>
                  <span className="font-medium ml-2">
                    {user.dob ? formatDate(user.dob) : 'Chưa cập nhật'}
                  </span>
                </div>
              </div>
            </div>

            {/* Address */}
            {user.address && (
              <div>
                <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <MapPin className="w-5 h-5 mr-2 text-blue-600" />
                  Địa chỉ
                </h4>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-gray-700">{user.address}</p>
                </div>
              </div>
            )}

            {/* Role & Permissions */}
            <div>
              <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                <Shield className="w-5 h-5 mr-2 text-blue-600" />
                Vai trò & Quyền hạn
              </h4>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="mb-3">
                  <span className="text-gray-600">Vai trò:</span>
                  <span className={`ml-2 px-3 py-1 text-sm font-medium rounded-full ${getRoleBadgeColor(user.role?.name)}`}>
                    {user.role?.name}
                  </span>
                </div>
                {user.role?.description && (
                  <div className="text-sm text-gray-600">
                    {user.role.description}
                  </div>
                )}
              </div>
            </div>

            {/* Account Info */}
            <div>
              <h4 className="text-lg font-semibold text-gray-800 mb-3">
                Thông tin tài khoản
              </h4>
              <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Ngày tạo:</span>
                  <span className="font-medium">{formatDate(user.created_at)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Cập nhật lần cuối:</span>
                  <span className="font-medium">{formatDate(user.updated_at)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Trạng thái:</span>
                  <Badge text={user.is_active ? 'Hoạt động' : 'Ngừng hoạt động'} />
                </div>
              </div>
            </div>

            {/* Additional Info based on Role */}
            {user.role?.name === 'Doctor' && user.doctor && (
              <div>
                <h4 className="text-lg font-semibold text-gray-800 mb-3">
                  Thông tin Bác sĩ
                </h4>
                <div className="bg-blue-50 rounded-lg p-4 space-y-2">
                  <div>
                    <span className="text-gray-600">Chuyên khoa:</span>
                    <span className="font-medium ml-2">{user.doctor.specialty}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Số chứng chỉ:</span>
                    <span className="font-medium ml-2">{user.doctor.license_number}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Kinh nghiệm:</span>
                    <span className="font-medium ml-2">{user.doctor.years_of_experience} năm</span>
                  </div>
                </div>
              </div>
            )}

            {user.role?.name === 'Patient' && user.patient && (
              <div>
                <h4 className="text-lg font-semibold text-gray-800 mb-3">
                  Thông tin Bệnh nhân
                </h4>
                <div className="bg-green-50 rounded-lg p-4 space-y-2">
                  <div>
                    <span className="text-gray-600">Nhóm máu:</span>
                    <span className="font-medium ml-2">{user.patient.blood_type || 'Chưa xác định'}</span>
                  </div>
                  {user.patient.allergies && (
                    <div>
                      <span className="text-gray-600">Dị ứng:</span>
                      <span className="font-medium ml-2">{user.patient.allergies}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </Modal>
  );
}