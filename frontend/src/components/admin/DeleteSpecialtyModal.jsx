import React, { useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import specialtyService from '../../services/specialtyService';

export default function DeleteSpecialtyModal({ specialty, onClose, onSuccess }) {
  const [loading, setLoading] = useState(false);

  const handleDelete = async () => {
    setLoading(true);
    try {
      await specialtyService.delete(specialty.id);
      onSuccess();
      onClose();
    } catch (error) {
      alert(`Lỗi: ${error.response?.data?.message || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="flex justify-between items-center p-6 border-b">
          <h2 className="text-xl font-bold">Xác nhận xóa</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-900"
            disabled={loading}
          >
            <X size={24} />
          </button>
        </div>

        <div className="p-6 space-y-4">
          <div className="flex gap-3">
            <AlertTriangle className="text-red-600 flex-shrink-0" size={24} />
            <div>
              <p className="font-semibold text-gray-900">Xóa chuyên khoa "{specialty.name}"?</p>
              <p className="text-sm text-gray-600 mt-1">Hành động này không thể được hoàn tác.</p>
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-900 hover:bg-gray-50 transition"
              disabled={loading}
            >
              Hủy
            </button>
            <button
              onClick={handleDelete}
              className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition disabled:opacity-50"
              disabled={loading}
            >
              {loading ? 'Đang xóa...' : 'Xóa'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
