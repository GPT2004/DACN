import React, { useState, useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { createStock } from '../../services/medicineService';

export default function AddStockModal({ medicine, onClose, onSuccess }) {
  const [quantity, setQuantity] = useState('');
  const [batchNumber, setBatchNumber] = useState('');
  const [expiryDate, setExpiryDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const qtyRef = useRef(null);

  useEffect(() => {
    // autofocus quantity when modal opens
    qtyRef.current?.focus();
  }, []);

  if (!medicine) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    const q = parseInt(String(quantity).replace(/[^0-9]/g, ''), 10);
    if (Number.isNaN(q) || q <= 0) {
      setError('Vui lòng nhập số lượng hợp lệ (> 0)');
      return;
    }
    if (!batchNumber.trim()) {
      setError('Vui lòng nhập lô hàng');
      return;
    }
    if (!expiryDate) {
      setError('Vui lòng chọn ngày hết hạn');
      return;
    }
    setError('');
    try {
      setLoading(true);
      await createStock({ 
        medicine_id: medicine.id, 
        quantity: q, 
        batch_number: batchNumber.trim(),
        expiry_date: expiryDate
      });
      onSuccess && onSuccess();
      onClose && onClose();
    } catch (err) {
      setError(err.response?.data?.message || 'Không thể thêm tồn kho');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div className="sticky top-0 flex items-center justify-between p-6 border-b border-gray-200 bg-white">
          <h2 className="text-xl font-bold text-gray-900">Nhập kho: {medicine.name}</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 transition">
            <X size={24} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && <div className="text-sm text-red-600">{error}</div>}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Số lượng <span className="text-red-500">*</span></label>
            <input
              ref={qtyRef}
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              placeholder="Nhập số lượng muốn thêm"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Lô hàng <span className="text-red-500">*</span></label>
            <input
              type="text"
              value={batchNumber}
              onChange={(e) => setBatchNumber(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              placeholder="Ví dụ: LOT001, BATCH-2024"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Ngày hết hạn <span className="text-red-500">*</span></label>
            <input
              type="date"
              value={expiryDate}
              onChange={(e) => setExpiryDate(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
            />
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition font-medium"
            >
              Hủy
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition font-medium"
            >
              {loading ? 'Đang xử lý...' : 'Thêm vào kho'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
