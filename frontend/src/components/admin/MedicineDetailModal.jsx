import React from 'react';
import { X } from 'lucide-react';

export default function MedicineDetailModal({ medicine, onClose }) {
  if (!medicine) return null;

  const stock = medicine.total_stock || 0;
  const status = stock === 0 ? 'Hết hàng' : stock < 10 ? 'Sắp hết' : 'Có sẵn';
  const statusColor = stock === 0 ? 'bg-red-100 text-red-800' : stock < 10 ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800';

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-screen overflow-y-auto">
        <div className="sticky top-0 flex items-center justify-between p-6 border-b border-gray-200 bg-white">
          <h2 className="text-xl font-bold text-gray-900">Chi tiết thuốc</h2>
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
                <label className="text-xs font-medium text-gray-500">Tên thuốc</label>
                <p className="text-gray-900 font-medium">{medicine.name || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Mã thuốc</label>
                <p className="text-gray-900">{medicine.code || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Mô tả</label>
                <p className="text-gray-900">{medicine.description || 'Không có mô tả'}</p>
              </div>
            </div>
          </div>

          {/* Thông tin kho */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b">Thông tin kho</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <div>
                  <label className="text-xs font-medium text-gray-500">Số lượng trong kho</label>
                  <p className="text-gray-900 font-semibold text-lg">{stock} {medicine.unit || 'cái'}</p>
                </div>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Trạng thái</label>
                <p className="mt-1">
                  <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${statusColor}`}>
                    {status}
                  </span>
                </p>
              </div>
            </div>
          </div>

          {/* Thông tin giá */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b">Thông tin giá</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-gray-500">Đơn vị tính</label>
                <p className="text-gray-900">{medicine.unit || 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Giá (đ)</label>
                <p className="text-gray-900 font-semibold text-lg">{medicine.price?.toLocaleString('vi-VN') || '0'} đ</p>
              </div>
            </div>
          </div>

          {/* Thông tin hệ thống */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-4 pb-2 border-b">Thông tin hệ thống</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-gray-500">Ngày tạo</label>
                <p className="text-gray-900">{medicine.created_at ? new Date(medicine.created_at).toLocaleString('vi-VN') : 'N/A'}</p>
              </div>
              <div>
                <label className="text-xs font-medium text-gray-500">Ngày cập nhật</label>
                <p className="text-gray-900">{medicine.updated_at ? new Date(medicine.updated_at).toLocaleString('vi-VN') : 'N/A'}</p>
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
