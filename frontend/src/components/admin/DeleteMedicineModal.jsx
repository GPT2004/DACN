import React, { useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import { deleteMedicine } from '../../services/medicineService';

export default function DeleteMedicineModal({ medicine, onClose, onSuccess }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleDelete = async () => {
    try {
      setLoading(true);
      await deleteMedicine(medicine.id);
      onSuccess();
      onClose();
    } catch (err) {
      setError(err.response?.data?.message || 'Xóa thất bại');
    } finally {
      setLoading(false);
    }
  };

  if (!medicine) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-sm w-full mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-red-100 rounded-full">
              <AlertTriangle className="text-red-600" size={24} />
            </div>
            <h2 className="text-xl font-bold text-gray-900">Xác nhận xóa</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 transition"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          {error && (
            <div className="alert alert-error">
              {error}
            </div>
          )}

          <p className="text-gray-700">
            Bạn có chắc chắn muốn xóa thuốc <span className="font-semibold">{medicine.name}</span>?
          </p>

          {medicine.dosage && (
            <p className="text-sm text-gray-500">
              Liều lượng: <span className="font-semibold">{medicine.dosage}</span>
            </p>
          )}

          {medicine.price && (
            <p className="text-sm text-gray-500">
              Giá: <span className="font-semibold">{new Intl.NumberFormat('vi-VN', { style: 'currency', currency: 'VND' }).format(medicine.price)}</span>
            </p>
          )}

          <div className="bg-red-50 border border-red-200 rounded-lg p-3">
            <p className="text-sm text-red-800">
              ⚠️ Hành động này không thể hoàn tác. Thuốc sẽ bị xóa hoàn toàn.
            </p>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex gap-3 p-6 border-t border-gray-200">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition font-medium"
          >
            Hủy
          </button>
          <button
            onClick={handleDelete}
            disabled={loading}
            className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-400 transition font-medium"
          >
            {loading ? 'Đang xóa...' : 'Xóa'}
          </button>
        </div>
      </div>
    </div>
  );
}
