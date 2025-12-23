/* eslint-disable no-console */
/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-unused-vars */
// frontend/src/components/admin/Reports/AppointmentReport.jsx
import React, { useState, useEffect } from 'react';
import { Calendar, TrendingUp, Clock, CheckCircle, XCircle } from 'lucide-react';
import { reportService } from '../../../services/reportService';
import { formatNumber } from '../../../utils/helpers';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Loader from '../../common/Loader';

export default function AppointmentReport() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [dateRange, setDateRange] = useState('thisMonth');
  const [status, setStatus] = useState('');

  useEffect(() => {
    fetchAppointmentReport();
  }, [dateRange, status]);

  const fetchAppointmentReport = async () => {
    try {
      setLoading(true);
      const range = getDateRangeFromPreset(dateRange);
      const response = await reportService.getAppointmentReport(
        range.start,
        range.end,
        status
      );
      setData(response.data);
    } catch (error) {
      console.error('Fetch appointment report error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDateRangeFromPreset = (preset) => {
    const today = new Date();
    let start, end;
    
    const getLocalDate = (d) => {
      const y = d.getFullYear();
      const m = String(d.getMonth() + 1).padStart(2, '0');
      const day = String(d.getDate()).padStart(2, '0');
      return `${y}-${m}-${day}`;
    };

    switch (preset) {
      case 'today':
        start = end = getLocalDate(today);
        break;
      case 'thisWeek':
        const weekStart = new Date(today.setDate(today.getDate() - today.getDay()));
        start = getLocalDate(weekStart);
        end = getLocalDate(new Date());
        break;
      case 'thisMonth':
        start = getLocalDate(new Date(today.getFullYear(), today.getMonth(), 1));
        end = getLocalDate(new Date());
        break;
      case 'lastMonth':
        start = getLocalDate(new Date(today.getFullYear(), today.getMonth() - 1, 1));
        end = getLocalDate(new Date(today.getFullYear(), today.getMonth(), 0));
        break;
      default:
        start = end = getLocalDate(today);
    }

    return { start, end };
  };

  if (loading) {
    return <Loader />;
  }

  const COLORS = ['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#8b5cf6', '#6b7280'];

  const statusData = [
    { name: 'Hoàn thành', value: data?.summary?.completed || 0, color: '#22c55e' },
    { name: 'Đã xác nhận', value: data?.summary?.confirmed || 0, color: '#3b82f6' },
    { name: 'Chờ xác nhận', value: data?.summary?.pending || 0, color: '#eab308' },
    { name: 'Đã hủy', value: data?.summary?.cancelled || 0, color: '#ef4444' },
    { name: 'Không đến', value: data?.summary?.noShow || 0, color: '#6b7280' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <h2 className="text-2xl font-bold text-gray-800">Báo cáo Lịch hẹn</h2>

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
              <option value="today">Hôm nay</option>
              <option value="thisWeek">Tuần này</option>
              <option value="thisMonth">Tháng này</option>
              <option value="lastMonth">Tháng trước</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Trạng thái
            </label>
            <select
              className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              value={status}
              onChange={(e) => setStatus(e.target.value)}
            >
              <option value="">Tất cả</option>
              <option value="PENDING">Chờ xác nhận</option>
              <option value="CONFIRMED">Đã xác nhận</option>
              <option value="COMPLETED">Hoàn thành</option>
              <option value="CANCELLED">Đã hủy</option>
              <option value="NO_SHOW">Không đến</option>
            </select>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-5 gap-4">
        <div className="bg-blue-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Tổng lịch hẹn</span>
            <Calendar className="w-6 h-6 text-blue-600" />
          </div>
          <div className="text-3xl font-bold text-blue-600">
            {formatNumber(data?.summary?.total || 0)}
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Hoàn thành</span>
            <CheckCircle className="w-6 h-6 text-green-600" />
          </div>
          <div className="text-3xl font-bold text-green-600">
            {formatNumber(data?.summary?.completed || 0)}
          </div>
        </div>

        <div className="bg-yellow-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Chờ xác nhận</span>
            <Clock className="w-6 h-6 text-yellow-600" />
          </div>
          <div className="text-3xl font-bold text-yellow-600">
            {formatNumber(data?.summary?.pending || 0)}
          </div>
        </div>

        <div className="bg-red-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Đã hủy</span>
            <XCircle className="w-6 h-6 text-red-600" />
          </div>
          <div className="text-3xl font-bold text-red-600">
            {formatNumber(data?.summary?.cancelled || 0)}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-sm text-gray-600 mb-2">Tỷ lệ hoàn thành</div>
          <div className="text-3xl font-bold text-gray-800">
            {data?.summary?.total > 0
              ? ((data.summary.completed / data.summary.total) * 100).toFixed(1)
              : 0}%
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Status Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Phân bổ theo trạng thái</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={statusData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {statusData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Daily Trend */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Xu hướng theo ngày</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data?.dailyTrend || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" name="Số lịch hẹn" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Appointments by Source */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Nguồn đặt lịch</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data?.bySource || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="source" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" name="Số lượng" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top Specialties and Doctors */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Top Chuyên khoa</h3>
          <div className="space-y-3">
            {data?.topSpecialties?.map((specialty, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium text-gray-900">{specialty.name}</div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-sm text-gray-600">
                    {formatNumber(specialty.count)} lịch
                  </div>
                  <div className="text-sm font-semibold text-blue-600">
                    {((specialty.count / data.summary.total) * 100).toFixed(1)}%
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
                  <div className="text-sm text-gray-600">{doctor.specialty}</div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-semibold text-green-600">
                    {formatNumber(doctor.count)} lịch
                  </div>
                  <div className="text-xs text-gray-500">
                    {doctor.completionRate}% hoàn thành
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Time Slot Analysis */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Phân bổ theo khung giờ</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data?.byTimeSlot || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timeSlot" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" name="Số lịch hẹn" fill="#22c55e" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}