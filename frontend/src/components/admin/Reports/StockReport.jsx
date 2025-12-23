/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
// frontend/src/components/admin/Reports/StockReport.jsx
import React, { useState, useEffect } from 'react';
import { Package, AlertTriangle, TrendingDown, Calendar } from 'lucide-react';
import { reportService } from '../../../services/reportService';
import { formatNumber, formatCurrency, formatDate } from '../../../utils/helpers';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Loader from '../../common/Loader';

export default function StockReport() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchStockReport();
  }, []);

  const fetchStockReport = async () => {
    try {
      setLoading(true);
      const response = await reportService.getStockReport();
      setData(response.data);
    } catch (error) {
      console.error('Fetch stock report error:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <Loader />;
  }

  const COLORS = ['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#8b5cf6'];

  const stockLevelData = [
    { name: 'Đầy đủ', value: data?.summary?.adequate || 0, color: '#22c55e' },
    { name: 'Sắp hết', value: data?.summary?.low || 0, color: '#eab308' },
    { name: 'Hết hàng', value: data?.summary?.outOfStock || 0, color: '#ef4444' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <h2 className="text-2xl font-bold text-gray-800">Báo cáo Tồn kho</h2>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-blue-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Tổng thuốc</span>
            <Package className="w-6 h-6 text-blue-600" />
          </div>
          <div className="text-3xl font-bold text-blue-600">
            {formatNumber(data?.summary?.totalMedicines || 0)}
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-6">
          <div className="text-sm text-gray-600 mb-2">Giá trị tồn kho</div>
          <div className="text-2xl font-bold text-green-600">
            {formatCurrency(data?.summary?.totalValue || 0)}
          </div>
        </div>

        <div className="bg-yellow-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Sắp hết</span>
            <AlertTriangle className="w-6 h-6 text-yellow-600" />
          </div>
          <div className="text-3xl font-bold text-yellow-600">
            {formatNumber(data?.summary?.low || 0)}
          </div>
        </div>

        <div className="bg-red-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Sắp hết hạn</span>
            <Calendar className="w-6 h-6 text-red-600" />
          </div>
          <div className="text-3xl font-bold text-red-600">
            {formatNumber(data?.summary?.expiring || 0)}
          </div>
        </div>
      </div>

      {/* Alerts */}
      {(data?.lowStockItems?.length > 0 || data?.expiringItems?.length > 0) && (
        <div className="space-y-4">
          {data.lowStockItems?.length > 0 && (
            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
              <div className="flex">
                <AlertTriangle className="w-5 h-5 text-yellow-400" />
                <div className="ml-3 flex-1">
                  <h3 className="text-sm font-medium text-yellow-800">
                    Cảnh báo: {data.lowStockItems.length} thuốc sắp hết hàng
                  </h3>
                  <div className="mt-2 text-sm text-yellow-700">
                    <ul className="list-disc list-inside">
                      {data.lowStockItems.slice(0, 5).map((item) => (
                        <li key={item.id}>
                          {item.name}: Còn {item.stock} {item.unit}
                        </li>
                      ))}
                      {data.lowStockItems.length > 5 && (
                        <li>Và {data.lowStockItems.length - 5} thuốc khác...</li>
                      )}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {data.expiringItems?.length > 0 && (
            <div className="bg-red-50 border-l-4 border-red-400 p-4">
              <div className="flex">
                <Calendar className="w-5 h-5 text-red-400" />
                <div className="ml-3 flex-1">
                  <h3 className="text-sm font-medium text-red-800">
                    Cảnh báo: {data.expiringItems.length} lô thuốc sắp hết hạn
                  </h3>
                  <div className="mt-2 text-sm text-red-700">
                    <ul className="list-disc list-inside">
                      {data.expiringItems.slice(0, 5).map((item) => (
                        <li key={item.id}>
                          {item.medicine_name} (Lô: {item.batch_number}) - HSD: {formatDate(item.expiry_date)}
                        </li>
                      ))}
                      {data.expiringItems.length > 5 && (
                        <li>Và {data.expiringItems.length - 5} lô khác...</li>
                      )}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Stock Level Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Phân bổ Tồn kho</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={stockLevelData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {stockLevelData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Stock by Category */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Tồn kho theo Loại</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data?.byCategory || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" name="Số lượng" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Stock Value by Category */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Giá trị Tồn kho theo Loại</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data?.valueByCategory || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="category" />
            <YAxis />
            <Tooltip formatter={(value) => formatCurrency(value)} />
            <Legend />
            <Bar dataKey="value" name="Giá trị" fill="#22c55e" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top Stock Items */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Top Thuốc tồn kho cao</h3>
          <div className="space-y-3">
            {data?.topStockItems?.map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{item.name}</div>
                  <div className="text-sm text-gray-600">{item.category}</div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-blue-600">
                    {formatNumber(item.stock)} {item.unit}
                  </div>
                  <div className="text-sm text-gray-600">
                    {formatCurrency(item.value)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Thuốc cần nhập thêm</h3>
          <div className="space-y-3">
            {data?.needRestock?.map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{item.name}</div>
                  <div className="text-sm text-gray-600">{item.category}</div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-red-600">
                    {formatNumber(item.stock)} {item.unit}
                  </div>
                  <div className="text-xs text-yellow-700">
                    Cần nhập: {formatNumber(item.suggested)} {item.unit}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Stock Movement */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Biến động Tồn kho (30 ngày qua)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data?.stockMovement || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="in" name="Nhập" fill="#22c55e" />
            <Bar dataKey="out" name="Xuất" fill="#ef4444" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Stock Details Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold">Chi tiết Tồn kho</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Mã thuốc
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Tên thuốc
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Loại
                </th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                  Tồn kho
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Đơn giá
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Giá trị
                </th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                  Trạng thái
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data?.stockDetails?.map((item) => (
                <tr key={item.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {item.code}
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm font-medium text-gray-900">
                      {item.name}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                      {item.category}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    <div className="text-sm font-medium text-gray-900">
                      {formatNumber(item.stock)} {item.unit}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                    {formatCurrency(item.unit_price)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-green-600">
                    {formatCurrency(item.total_value)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    {item.stock === 0 ? (
                      <span className="px-2 py-1 text-xs font-semibold bg-red-100 text-red-800 rounded">
                        Hết hàng
                      </span>
                    ) : item.stock < 20 ? (
                      <span className="px-2 py-1 text-xs font-semibold bg-yellow-100 text-yellow-800 rounded">
                        Sắp hết
                      </span>
                    ) : (
                      <span className="px-2 py-1 text-xs font-semibold bg-green-100 text-green-800 rounded">
                        Đầy đủ
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}