import React from 'react';
import { X } from 'lucide-react';

export default function DoctorDetailModal({ doctor, onClose }) {
  if (!doctor) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-screen overflow-y-auto">
        <div className="sticky top-0 flex items-center justify-between p-6 border-b border-gray-200 bg-white">
          <h2 className="text-xl font-bold text-gray-900">Chi tiết bác sĩ</h2>
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
                <label className="text-xs font-medium text-gray-500">Tên bác sĩ</label>
                <p className="text-gray-900 font-medium">{doctor.user?.full_name || doctor.full_name || doctor.fullName || doctor.name || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Email</label>
                <p className="text-gray-900">{doctor.user?.email || doctor.email || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Chuyên môn</label>
                <p className="mt-1">
                  <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-semibold">
                    {doctor.specialties && doctor.specialties.length > 0 ? doctor.specialties.join(', ') : doctor.specialty || 'Chưa cập nhật'}
                  </span>
                </p>
              </div>
            </div>
          </div>

          {/* Thông tin liên hệ */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b">Thông tin liên hệ</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-gray-500">Số điện thoại</label>
                <p className="text-gray-900">{doctor.user?.phone || doctor.phone_number || doctor.phoneNumber || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Địa chỉ</label>
                <p className="text-gray-900">{doctor.address || 'N/A'}</p>
              </div>
            </div>
          </div>

          {/* Thông tin bổ sung */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b">Thông tin bổ sung</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-gray-500">Giấy phép hành nghề</label>
                <p className="text-gray-900">{doctor.license_number || doctor.licenseNumber || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Kinh nghiệm (năm)</label>
                <p className="text-gray-900">{doctor.years_of_experience || doctor.yearsOfExperience || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Bệnh viện</label>
                <p className="text-gray-900">{doctor.hospital || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Ngày tạo</label>
                <p className="text-gray-900">{doctor.created_at ? new Date(doctor.created_at).toLocaleString('vi-VN') : 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Ngày cập nhật</label>
                <p className="text-gray-900">{doctor.updated_at ? new Date(doctor.updated_at).toLocaleString('vi-VN') : 'N/A'}</p>
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
