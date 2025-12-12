import React from 'react';
import { X } from 'lucide-react';

export default function SpecialtyDetailModal({ specialty, onClose }) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="flex justify-between items-center p-6 border-b">
          <h2 className="text-xl font-bold">Chi tiết chuyên khoa</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-900"
          >
            <X size={24} />
          </button>
        </div>

        <div className="p-6 space-y-4">
          {specialty.image_url && (
            <div className="mb-4">
              <img src={specialty.image_url} alt={specialty.name} className="w-full h-32 rounded-lg object-cover" />
            </div>
          )}

          <div>
            <p className="text-sm text-gray-600">Tên Chuyên Khoa</p>
            <p className="font-semibold text-gray-900">{specialty.name}</p>
          </div>

          {specialty.slug && (
            <div>
              <p className="text-sm text-gray-600">Slug</p>
              <p className="font-semibold text-gray-900">{specialty.slug}</p>
            </div>
          )}

          {specialty.description && (
            <div>
              <p className="text-sm text-gray-600">Mô Tả</p>
              <p className="text-gray-900">{specialty.description}</p>
            </div>
          )}

          {specialty.created_at && (
            <div>
              <p className="text-sm text-gray-600">Ngày Tạo</p>
              <p className="text-gray-900">{new Date(specialty.created_at).toLocaleString('vi-VN')}</p>
            </div>
          )}

          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition mt-4"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
}
