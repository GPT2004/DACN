/* eslint-disable no-console */
/* eslint-disable react-hooks/exhaustive-deps */
// frontend/src/components/admin/Reports/RevenueReport.jsx
import React, { useState, useEffect } from 'react';
import { DollarSign, TrendingUp, TrendingDown, Download } from 'lucide-react';
import { reportService } from '../../../services/reportService';
import { formatCurrency, formatNumber } from '../../../utils/helpers';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Button from '../../common/Button';
import Loader from '../../common/Loader';

export default function RevenueReport() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [dateRange, setDateRange] = useState('thisMonth');
  const [groupBy, setGroupBy] = useState('day');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    fetchRevenueReport();
  }, [dateRange, groupBy]);

  const fetchRevenueReport = async () => {
    try {
      setLoading(true);
      let start, end;

      if (dateRange === 'custom') {
        start = startDate;
        end = endDate;
      } else {
        const range = getDateRangeFromPreset(dateRange);
        start = range.start;
        end = range.end;
      }

      const response = await reportService.getRevenueReport(start, end, groupBy);
      setData(response.data);
    } catch (error) {
      console.error('Fetch revenue report error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDateRangeFromPreset = (preset) => {
    const today = new Date();
    let start, end;

    switch (preset) {
      case 'today':
        start = end = today.toISOString().split('T')[0];
        break;
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

  const handleExport = async () => {
    try {
      const response = await reportService.exportReport('revenue', 'xlsx', {
        startDate: dateRange === 'custom' ? startDate : getDateRangeFromPreset(dateRange).start,
        endDate: dateRange === 'custom' ? endDate : getDateRangeFromPreset(dateRange).end,
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `revenue-report-${new Date().getTime()}.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Export error:', error);
    }
  };

  if (loading) {
    return <Loader />;
  }

  const calculateGrowth = () => {
    if (!data?.comparison) return 0;
    const { current, previous } = data.comparison;
    if (previous === 0) return 100;
    return (((current - previous) / previous) * 100).toFixed(1);
  };

  const growth = calculateGrowth();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Báo cáo Doanh thu</h2>
        <Button onClick={handleExport}>
          <Download className="w-4 h-4 mr-2" />
          Xuất báo cáo
        </Button>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Khoảng thời gian
            </label>
            <select
              className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
            >
              <option value="today">Hôm nay</option>
              <option value="thisWeek">Tuần này</option>
              <option value="thisMonth">Tháng này</option>
              <option value="lastMonth">Tháng trước</option>
              <option value="thisYear">Năm này</option>
              <option value="custom">Tùy chỉnh</option>
            </select>
          </div>

          {dateRange === 'custom' && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Từ ngày
                </label>
                <input
                  type="date"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Đến ngày
                </label>
                <input
                  type="date"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
            </>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Nhóm theo
            </label>
            <select
              className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              value={groupBy}
              onChange={(e) => setGroupBy(e.target.value)}
            >
              <option value="day">Ngày</option>
              <option value="week">Tuần</option>
              <option value="month">Tháng</option>
            </select>
          </div>
        </div>

        {dateRange === 'custom' && (
          <div className="mt-4">
            <Button onClick={fetchRevenueReport}>Áp dụng</Button>
          </div>
        )}
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between mb-2">
            <span className="text-blue-100">Tổng doanh thu</span>
            <DollarSign className="w-8 h-8 text-blue-100" />
          </div>
          <div className="text-3xl font-bold">
            {formatCurrency(data?.summary?.totalRevenue || 0)}
          </div>
          <div className="mt-2 flex items-center text-sm">
            {growth >= 0 ? (
              <>
                <TrendingUp className="w-4 h-4 mr-1" />
                <span>+{growth}%</span>
              </>
            ) : (
              <>
                <TrendingDown className="w-4 h-4 mr-1" />
                <span>{growth}%</span>
              </>
            )}
            <span className="ml-1 text-blue-100">so với kỳ trước</span>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-sm text-gray-600 mb-2">Số hóa đơn</div>
          <div className="text-3xl font-bold text-gray-800">
            {formatNumber(data?.summary?.totalInvoices || 0)}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-sm text-gray-600 mb-2">Giá trị TB/hóa đơn</div>
          <div className="text-3xl font-bold text-gray-800">
            {formatCurrency(data?.summary?.averageInvoiceValue || 0)}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-sm text-gray-600 mb-2">Đã thanh toán</div>
          <div className="text-3xl font-bold text-green-600">
            {formatCurrency(data?.summary?.paidAmount || 0)}
          </div>
        </div>
      </div>

      {/* Revenue Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Biểu đồ Doanh thu</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data?.chartData || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="period" />
            <YAxis />
            <Tooltip 
              formatter={(value) => formatCurrency(value)}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="revenue" 
              name="Doanh thu"
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="paid" 
              name="Đã thanh toán"
              stroke="#22c55e" 
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Revenue by Payment Method */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Doanh thu theo phương thức thanh toán</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data?.paymentMethodData || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="method" />
            <YAxis />
            <Tooltip formatter={(value) => formatCurrency(value)} />
            <Legend />
            <Bar dataKey="amount" name="Doanh thu" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top Revenue Sources */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Top Dịch vụ</h3>
          <div className="space-y-3">
            {data?.topServices?.map((service, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{service.name}</div>
                  <div className="text-sm text-gray-600">
                    {formatNumber(service.count)} lượt
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-blue-600">
                    {formatCurrency(service.revenue)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Top Bác sĩ</h3>
          <div className="space-y-3">
            {data?.topDoctors?.map((doctor, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{doctor.name}</div>
                  <div className="text-sm text-gray-600">
                    {formatNumber(doctor.appointments)} lịch hẹn
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-green-600">
                    {formatCurrency(doctor.revenue)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}