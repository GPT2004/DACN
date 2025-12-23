/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
// frontend/src/components/admin/MedicineManagement/StockManager.jsx
import React, { useState, useEffect } from 'react';
import { Plus, Minus, AlertTriangle, Calendar, Package } from 'lucide-react';
import { medicineService } from '../../../services/medicineService';
import { formatCurrency, formatNumber, formatDate } from '../../../utils/helpers';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Badge from '../../common/Badge';
import Loader from '../../common/Loader';
import Modal from '../../common/Modal';

export default function StockManager() {
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState(null);
  const [lowStockAlerts, setLowStockAlerts] = useState([]);
  const [expiringMeds, setExpiringMeds] = useState([]);
  const [showAddStock, setShowAddStock] = useState(false);
  const [selectedMedicine, setSelectedMedicine] = useState(null);

  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      const [stocksRes, summaryRes, lowStockRes, expiringRes] = await Promise.all([
        medicineService.getAllStocks(),
        medicineService.getStockSummary(),
        medicineService.getLowStockAlerts(),
        medicineService.getExpiringMedicines(30)
      ]);

      setStocks(stocksRes.data.stocks || []);
      setSummary(summaryRes.data);
      setLowStockAlerts(lowStockRes.data.alerts || []);
      setExpiringMeds(expiringRes.data.medicines || []);
    } catch (error) {
      console.error('Fetch stocks error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddStock = (medicine) => {
    setSelectedMedicine(medicine);
    setShowAddStock(true);
  };

  if (loading) {
    return <Loader />;
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Tổng thuốc</p>
              <p className="text-2xl font-bold text-blue-600">
                {formatNumber(summary?.totalMedicines || 0)}
              </p>
            </div>
            <Package className="w-8 h-8 text-blue-600" />
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Tổng giá trị</p>
              <p className="text-2xl font-bold text-green-600">
                {formatCurrency(summary?.totalValue || 0)}
              </p>
            </div>
            <Package className="w-8 h-8 text-green-600" />
          </div>
        </div>

        <div className="bg-yellow-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Sắp hết hàng</p>
              <p className="text-2xl font-bold text-yellow-600">
                {lowStockAlerts.length}
              </p>
            </div>
            <AlertTriangle className="w-8 h-8 text-yellow-600" />
          </div>
        </div>

        <div className="bg-red-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Sắp hết hạn</p>
              <p className="text-2xl font-bold text-red-600">
                {expiringMeds.length}
              </p>
            </div>
            <Calendar className="w-8 h-8 text-red-600" />
          </div>
        </div>
      </div>

      {/* Alerts */}
      {lowStockAlerts.length > 0 && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
          <div className="flex">
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">
                Cảnh báo: Có {lowStockAlerts.length} thuốc sắp hết hàng
              </h3>
              <div className="mt-2 text-sm text-yellow-700">
                <ul className="list-disc list-inside">
                  {lowStockAlerts.slice(0, 3).map((alert) => (
                    <li key={alert.medicine_id}>
                      {alert.medicine_name}: Còn {alert.total_stock} {alert.unit}
                    </li>
                  ))}
                  {lowStockAlerts.length > 3 && (
                    <li>Và {lowStockAlerts.length - 3} thuốc khác...</li>
                  )}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {expiringMeds.length > 0 && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <Calendar className="w-5 h-5 text-red-400" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                Cảnh báo: Có {expiringMeds.length} lô thuốc sắp hết hạn (trong 30 ngày)
              </h3>
              <div className="mt-2 text-sm text-red-700">
                <ul className="list-disc list-inside">
                  {expiringMeds.slice(0, 3).map((med) => (
                    <li key={med.id}>
                      {med.medicine_name} (Lô: {med.batch_number}) - HSD: {formatDate(med.expiry_date)}
                    </li>
                  ))}
                  {expiringMeds.length > 3 && (
                    <li>Và {expiringMeds.length - 3} lô khác...</li>
                  )}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Stock Table */}
      <div className="bg-white rounded-lg shadow">
        <div className="p-4 border-b">
          <h3 className="text-lg font-semibold">Quản lý Tồn kho</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Thuốc
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Số lô
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Số lượng
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Giá nhập
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Ngày nhập
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Hạn sử dụng
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Trạng thái
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Thao tác
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {stocks.map((stock) => {
                const daysToExpiry = Math.ceil(
                  (new Date(stock.expiry_date) - new Date()) / (1000 * 60 * 60 * 24)
                );
                const isExpiringSoon = daysToExpiry <= 30 && daysToExpiry > 0;
                const isExpired = daysToExpiry <= 0;

                return (
                  <tr key={stock.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-900">
                        {stock.medicine?.name}
                      </div>
                      <div className="text-sm text-gray-500">
                        {stock.medicine?.code}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {stock.batch_number}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">
                        {formatNumber(stock.quantity)} {stock.medicine?.unit}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatCurrency(stock.unit_price)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(stock.received_date)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">
                        {formatDate(stock.expiry_date)}
                      </div>
                      {isExpiringSoon && (
                        <div className="text-xs text-yellow-600">
                          Còn {daysToExpiry} ngày
                        </div>
                      )}
                      {isExpired && (
                        <div className="text-xs text-red-600">
                          Đã hết hạn
                        </div>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {isExpired ? (
                        <Badge text="Hết hạn" />
                      ) : isExpiringSoon ? (
                        <Badge text="Sắp hết hạn" />
                      ) : stock.quantity < 20 ? (
                        <Badge text="Sắp hết" />
                      ) : (
                        <Badge text="Bình thường" />
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <Button onClick={() => handleAddStock(stock.medicine)}>
                        <Plus className="w-4 h-4" />
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {stocks.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500">Không có dữ liệu tồn kho</p>
            </div>
          )}
        </div>
      </div>

      {/* Add Stock Modal */}
      {showAddStock && (
        <AddStockModal
          medicine={selectedMedicine}
          onClose={() => {
            setShowAddStock(false);
            setSelectedMedicine(null);
          }}
          onSuccess={fetchAllData}
        />
      )}
    </div>
  );
}

// Add Stock Modal Component
function AddStockModal({ medicine, onClose, onSuccess }) {
  const [formData, setFormData] = useState({
    batch_number: '',
    quantity: '',
    unit_price: '',
    received_date: new Date().toISOString().split('T')[0],
    expiry_date: '',
    supplier_id: '',
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      setLoading(true);
      await medicineService.createStock({
        medicine_id: medicine.id,
        ...formData,
      });
      onSuccess();
      onClose();
    } catch (error) {
      console.error('Add stock error:', error);
      alert('Có lỗi xảy ra!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal isOpen={!!medicine} onClose={onClose}>
      <div className="bg-white rounded-lg p-6 max-w-md w-full">
        <h3 className="text-lg font-semibold mb-4">
          Nhập kho: {medicine?.name}
        </h3>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Số lô <span className="text-red-500">*</span>
            </label>
            <Input
              value={formData.batch_number}
              onChange={(e) => setFormData({ ...formData, batch_number: e.target.value })}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Số lượng <span className="text-red-500">*</span>
            </label>
            <Input
              type="number"
              value={formData.quantity}
              onChange={(e) => setFormData({ ...formData, quantity: e.target.value })}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Giá nhập (VNĐ) <span className="text-red-500">*</span>
            </label>
            <Input
              type="number"
              value={formData.unit_price}
              onChange={(e) => setFormData({ ...formData, unit_price: e.target.value })}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Ngày nhập <span className="text-red-500">*</span>
            </label>
            <Input
              type="date"
              value={formData.received_date}
              onChange={(e) => setFormData({ ...formData, received_date: e.target.value })}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Hạn sử dụng <span className="text-red-500">*</span>
            </label>
            <Input
              type="date"
              value={formData.expiry_date}
              onChange={(e) => setFormData({ ...formData, expiry_date: e.target.value })}
              required
            />
          </div>

          <div className="flex gap-3 pt-4">
            <Button onClick={onClose} disabled={loading}>
              Hủy
            </Button>
            <Button type="submit" disabled={loading}>
              {loading ? 'Đang xử lý...' : 'Nhập kho'}
            </Button>
          </div>
        </form>
      </div>
    </Modal>
  );
}