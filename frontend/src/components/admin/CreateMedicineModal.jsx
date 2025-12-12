import React, { useEffect, useState, useRef } from 'react';
import { X } from 'lucide-react';
import { createMedicine, createStock } from '../../services/medicineService';

export default function CreateMedicineModal({ onClose, onSuccess }) {
  const [name, setName] = useState('');
  const [unit, setUnit] = useState('');
  const [price, setPrice] = useState('');
  const [code, setCode] = useState('');
  const [description, setDescription] = useState('');
  const [quantity, setQuantity] = useState('');
  const [batchNumber, setBatchNumber] = useState('');
  const [expiryDate, setExpiryDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const nameRef = useRef(null);

  useEffect(() => {
    // autofocus when opened
    nameRef.current?.focus();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) return setError('Vui lòng nhập tên thuốc');
    if (!unit.trim()) return setError('Vui lòng nhập đơn vị');
    const p = parseInt(String(price).replace(/[^0-9]/g, ''), 10);
    if (Number.isNaN(p) || p < 0) return setError('Giá không hợp lệ');

    // Validate stock fields
    const q = parseInt(String(quantity).replace(/[^0-9]/g, ''), 10);
    if (Number.isNaN(q) || q <= 0) return setError('Vui lòng nhập số lượng hợp lệ (> 0)');
    if (!batchNumber.trim()) return setError('Vui lòng nhập lô hàng');
    if (!expiryDate) return setError('Vui lòng chọn ngày hết hạn');

    setError('');
    try {
      setLoading(true);
      // Create medicine first
      const medicineRes = await createMedicine({
        name: name.trim(),
        unit: unit.trim(),
        price: p,
        code: code.trim(),
        description: description.trim(),
      });
      
      // Then create stock for the new medicine
      if (medicineRes?.data?.id || medicineRes?.id) {
        const medicineId = medicineRes?.data?.id || medicineRes?.id;
        await createStock({
          medicine_id: medicineId,
          quantity: q,
          batch_number: batchNumber.trim(),
          expiry_date: expiryDate,
        });
      }
      
      onSuccess && onSuccess();
      onClose && onClose();
    } catch (err) {
      setError(err.response?.data?.message || 'Không thể tạo thuốc/kho');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-screen overflow-y-auto">
        <div className="sticky top-0 flex items-center justify-between p-6 border-b border-gray-200 bg-white">
          <h2 className="text-xl font-bold text-gray-900">Thêm thuốc mới</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 transition">
            <X size={24} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && <div className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</div>}

          {/* Thông tin thuốc */}
          <div className="border-b pb-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Thông tin thuốc</h3>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Tên thuốc <span className="text-red-500">*</span></label>
              <input ref={nameRef} value={name} onChange={(e) => setName(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="Nhập tên thuốc" />
            </div>

            <div className="grid grid-cols-2 gap-3 mt-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Đơn vị <span className="text-red-500">*</span></label>
                <input value={unit} onChange={(e) => setUnit(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="ví dụ: viên" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Giá (đ)</label>
                <input type="number" value={price} onChange={(e) => setPrice(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="Giá" />
              </div>
            </div>

            <div className="mt-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Mã thuốc</label>
              <input value={code} onChange={(e) => setCode(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="Mã (tùy chọn)" />
            </div>

            <div className="mt-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Mô tả</label>
              <textarea value={description} onChange={(e) => setDescription(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg resize-none" rows="2" placeholder="Mô tả (tùy chọn)" />
            </div>
          </div>

          {/* Thông tin nhập kho */}
          <div className="border-b pb-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Nhập kho luôn</h3>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Số lượng <span className="text-red-500">*</span></label>
              <input type="number" value={quantity} onChange={(e) => setQuantity(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="Số lượng" />
            </div>

            <div className="mt-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Lô hàng <span className="text-red-500">*</span></label>
              <input value={batchNumber} onChange={(e) => setBatchNumber(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="ví dụ: LOT001" />
            </div>

            <div className="mt-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Ngày hết hạn <span className="text-red-500">*</span></label>
              <input type="date" value={expiryDate} onChange={(e) => setExpiryDate(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-lg" />
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition">Hủy</button>
            <button type="submit" disabled={loading} className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition">{loading ? 'Đang tạo...' : 'Tạo'}</button>
          </div>
        </form>
      </div>
    </div>
  );
}
