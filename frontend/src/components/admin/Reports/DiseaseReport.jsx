// frontend/src/components/admin/Reports/DiseaseReport.jsx
import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, AlertCircle } from 'lucide-react';
import { reportService } from '../../../services/reportService';
import { formatNumber } from '../../../utils/helpers';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Loader from '../../common/Loader';

export default function DiseaseReport() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [dateRange, setDateRange] = useState('thisMonth');
  const [limit, setLimit] = useState(10);

  useEffect(() => {
    fetchDiseaseReport();
  }, [dateRange, limit]);

  const fetchDiseaseReport = async () => {
    try {
      setLoading(true);
      const range = getDateRangeFromPreset(dateRange);
      const response = await reportService.getCommonDiseases(
        range.start,
        range.end,
        limit
      );
      setData(response.data);
    } catch (error) {
      console.error('Fetch disease report error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDateRangeFromPreset = (preset) => {
    const today = new Date();
    let start, end;

    switch (preset) {
      case 'thisWeek':
        const weekStart = new Date(today.setDate(today.getDate() - today.getDay()));
        start = weekStart.toISOString().split('T')[0];
        end = new Date().toISOString().split('T')[0];
        break;
      case 'thisMonth':
        start = new Date(today.getFullYear(), today.getMonth(), 1).toISOString().split('T')[0];
        end = new Date().toISOString().split('T')[0];
        break;
      case 'lastMonth':
        start = new Date(today.getFullYear(), today.getMonth() - 1, 1).toISOString().split('T')[0];
        end = new Date(today.getFullYear(), today.getMonth(), 0).toISOString().split('T')[0];
        break;
      case 'thisYear':
        start = new Date(today.getFullYear(), 0, 1).toISOString().split('T')[0];
        end = new Date().toISOString().split('T')[0];
        break;
      default:
        start = end = today.toISOString().split('T')[0];
    }

    return { start, end };
  };

  if (loading) {
    return <Loader />;
  }

  const COLORS = ['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <h2 className="text-2xl font-bold text-gray-800">Báo cáo Bệnh thường gặp</h2>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Khoảng thời gian
            </label>
            <select
              className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
            >
              <option value="thisWeek">Tuần này</option>
              <option value="thisMonth">Tháng này</option>
              <option value="lastMonth">Tháng trước</option>
              <option value="thisYear">Năm này</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Số lượng hiển thị
            </label>
            <select
              className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
            >
              <option value="5">Top 5</option>
              <option value="10">Top 10</option>
              <option value="15">Top 15</option>
              <option value="20">Top 20</option>
            </select>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-blue-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Tổng ca bệnh</span>
            <Activity className="w-6 h-6 text-blue-600" />
          </div>
          <div className="text-3xl font-bold text-blue-600">
            {formatNumber(data?.summary?.totalCases || 0)}
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Loại bệnh</span>
            <Activity className="w-6 h-6 text-green-600" />
          </div>
          <div className="text-3xl font-bold text-green-600">
            {formatNumber(data?.summary?.totalDiseases || 0)}
          </div>
        </div>

        <div className="bg-yellow-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Bệnh phổ biến nhất</span>
            <TrendingUp className="w-6 h-6 text-yellow-600" />
          </div>
          <div className="text-lg font-bold text-yellow-600">
            {data?.topDiseases?.[0]?.name || 'N/A'}
          </div>
          <div className="text-sm text-gray-600 mt-1">
            {formatNumber(data?.topDiseases?.[0]?.count || 0)} ca
          </div>
        </div>
      </div>

      {/* Top Diseases Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Top Bệnh thường gặp</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={data?.topDiseases || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" name="Số ca bệnh" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Disease Distribution by Category */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Phân bổ theo nhóm bệnh</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data?.byCategory || []}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="count"
              >
                {(data?.byCategory || []).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Phân bổ theo độ tuổi</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data?.byAgeGroup || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ageGroup" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" name="Số ca bệnh" fill="#22c55e" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Disease Details Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold">Chi tiết Bệnh</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  #
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Tên bệnh
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Nhóm bệnh
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Số ca
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Tỷ lệ
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Xu hướng
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data?.topDiseases?.map((disease, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {index + 1}
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm font-medium text-gray-900">
                      {disease.name}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                      {disease.category}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {formatNumber(disease.count)}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">
                      {((disease.count / data.summary.totalCases) * 100).toFixed(1)}%
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {disease.trend > 0 ? (
                      <span className="flex items-center text-sm text-green-600">
                        <TrendingUp className="w-4 h-4 mr-1" />
                        +{disease.trend}%
                      </span>
                    ) : disease.trend < 0 ? (
                      <span className="flex items-center text-sm text-red-600">
                        <TrendingUp className="w-4 h-4 mr-1 rotate-180" />
                        {disease.trend}%
                      </span>
                    ) : (
                      <span className="text-sm text-gray-500">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Seasonal Trends */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Xu hướng theo tháng</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data?.monthlyTrend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" name="Số ca bệnh" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Health Alerts */}
      {data?.alerts && data.alerts.length > 0 && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
          <div className="flex">
            <AlertCircle className="w-5 h-5 text-yellow-400" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">
                Cảnh báo sức khỏe
              </h3>
              <div className="mt-2 text-sm text-yellow-700">
                <ul className="list-disc list-inside">
                  {data.alerts.map((alert, index) => (
                    <li key={index}>{alert.message}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}