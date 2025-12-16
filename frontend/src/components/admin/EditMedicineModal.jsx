import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { updateMedicine, createStock, updateStock } from '../../services/medicineService';

export default function EditMedicineModal({ medicine, onClose, onSuccess }) {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    unit: '',
    price: '',
  });
  const [stockData, setStockData] = useState({
    quantity: '',
    batch_number: '',
    expiry_date: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (medicine) {
      setFormData({
        name: medicine.name || '',
        description: medicine.description || '',
        unit: medicine.unit || '',
        price: medicine.price || '',
      });
      if (medicine.stocks && medicine.stocks.length > 0) {
        const latestStock = medicine.stocks[0];
        const expiryDate = latestStock.expiry_date ? new Date(latestStock.expiry_date).toISOString().split('T')[0] : '';
        setStockData({
          quantity: '',
          batch_number: latestStock.batch_number || '',
          expiry_date: expiryDate,
          stock_id: latestStock.id,
        });
      } else {
        setStockData({
          quantity: '',
          batch_number: '',
          expiry_date: '',
        });
      }
    }
  }, [medicine]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError('');
  };

  const handleStockChange = (e) => {
    const { name, value } = e.target;
    setStockData(prev => ({ ...prev, [name]: value }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      setError('Vui lòng điền tên thuốc');
      return;
    }

    try {
      setLoading(true);
      // Update medicine info
      await updateMedicine(medicine.id, formData);

      // Handle stock: create new if quantity provided, update existing if only changing details
      if (stockData.quantity && String(stockData.quantity).trim() !== '') {
        const q = parseInt(String(stockData.quantity).replace(/[^0-9]/g, ''), 10);
        if (Number.isNaN(q) || q <= 0) {
          setError('Vui lòng nhập số lượng hợp lệ (> 0)');
          setLoading(false);
          return;
        }

        let expiryDate = stockData.expiry_date;
        if (!expiryDate) {
          const oneYearLater = new Date();
          oneYearLater.setFullYear(oneYearLater.getFullYear() + 1);
          expiryDate = oneYearLater.toISOString().split('T')[0];
        }

        await createStock({
          medicine_id: medicine.id,
          quantity: q,
          batch_number: stockData.batch_number.trim() || 'N/A',
          expiry_date: expiryDate,
        });
      } else if (stockData.stock_id && stockData.expiry_date) {
        // Update existing stock when only changing expiry/batch details
        await updateStock(stockData.stock_id, {
          batch_number: stockData.batch_number.trim() || 'N/A',
          expiry_date: stockData.expiry_date,
        });
      }

      onSuccess();
      onClose();
    } catch (err) {
      setError(err.response?.data?.message || 'Cập nhật thất bại');
    } finally {
      setLoading(false);
    }
  };

  if (!medicine) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 max-h-screen overflow-y-auto">
        <div className="sticky top-0 flex items-center justify-between p-6 border-b border-gray-200 bg-white">
          <h2 className="text-xl font-bold text-gray-900">Chỉnh sửa thuốc</h2>
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
              <input type="text" name="name" value={formData.name} onChange={handleChange} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="Nhập tên thuốc" />
            </div>

            <div className="mt-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Mô tả</label>
              <textarea name="description" value={formData.description} onChange={handleChange} className="w-full px-3 py-2 border border-gray-300 rounded-lg resize-none" rows="2" placeholder="Mô tả..." />
            </div>

            <div className="grid grid-cols-2 gap-3 mt-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Đơn vị</label>
                <input type="text" name="unit" value={formData.unit} onChange={handleChange} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="viên, gói..." />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Giá (đ)</label>
                <input type="number" name="price" value={formData.price} onChange={handleChange} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="Giá" />
              </div>
            </div>
          </div>

          {/* Nhập thêm kho (tùy chọn) */}
          <div className="border-b pb-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Nhập thêm kho (tùy chọn)</h3>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Số lượng</label>
              <input type="number" name="quantity" value={stockData.quantity} onChange={handleStockChange} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="Số lượng" />
            </div>

            <div className="mt-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Lô hàng</label>
              <input type="text" name="batch_number" value={stockData.batch_number} onChange={handleStockChange} className="w-full px-3 py-2 border border-gray-300 rounded-lg" placeholder="LOT001" />
            </div>

            <div className="mt-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Ngày hết hạn</label>
              <input type="date" name="expiry_date" value={stockData.expiry_date} onChange={handleStockChange} className="w-full px-3 py-2 border border-gray-300 rounded-lg" />
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition">Hủy</button>
            <button type="submit" disabled={loading} className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition">{loading ? 'Đang lưu...' : 'Lưu'}</button>
          </div>
        </form>
      </div>
    </div>
  );
}
