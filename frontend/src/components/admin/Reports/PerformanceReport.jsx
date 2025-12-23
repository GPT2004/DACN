/* eslint-disable no-console */
/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-unused-vars */
// frontend/src/components/admin/Reports/PerformanceReport.jsx
import React, { useState, useEffect } from 'react';
import { Award, TrendingUp, Star, Users } from 'lucide-react';
import { reportService } from '../../../services/reportService';
import { formatNumber, formatCurrency } from '../../../utils/helpers';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import Loader from '../../common/Loader';

export default function PerformanceReport() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [dateRange, setDateRange] = useState('thisMonth');
  const [selectedDoctor, setSelectedDoctor] = useState('');

  useEffect(() => {
    fetchPerformanceReport();
  }, [dateRange, selectedDoctor]);

  const fetchPerformanceReport = async () => {
    try {
      setLoading(true);
      const range = getDateRangeFromPreset(dateRange);
      const response = await reportService.getDoctorPerformance(
        range.start,
        range.end,
        selectedDoctor
      );
      setData(response.data);
    } catch (error) {
      console.error('Fetch performance report error:', error);
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <h2 className="text-2xl font-bold text-gray-800">Báo cáo Hiệu suất Bác sĩ</h2>

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
              Bác sĩ
            </label>
            <select
              className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              value={selectedDoctor}
              onChange={(e) => setSelectedDoctor(e.target.value)}
            >
              <option value="">Tất cả bác sĩ</option>
              {data?.doctors?.map((doctor) => (
                <option key={doctor.id} value={doctor.id}>
                  {doctor.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between mb-2">
            <span className="text-blue-100">Tổng lịch hẹn</span>
            <Users className="w-8 h-8 text-blue-100" />
          </div>
          <div className="text-3xl font-bold">
            {formatNumber(data?.summary?.totalAppointments || 0)}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-sm text-gray-600 mb-2">Bệnh nhân khám</div>
          <div className="text-3xl font-bold text-gray-800">
            {formatNumber(data?.summary?.totalPatients || 0)}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-sm text-gray-600 mb-2">Tỷ lệ hoàn thành</div>
          <div className="text-3xl font-bold text-green-600">
            {data?.summary?.completionRate || 0}%
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-sm text-gray-600 mb-2">Đánh giá TB</div>
          <div className="flex items-center">
            <div className="text-3xl font-bold text-yellow-500">
              {data?.summary?.averageRating || 0}
            </div>
            <Star className="w-6 h-6 text-yellow-500 ml-2" />
          </div>
        </div>
      </div>

      {/* Top Performers */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Award className="w-5 h-5 mr-2 text-yellow-500" />
          Top Bác sĩ xuất sắc
        </h3>
        <div className="grid grid-cols-3 gap-4">
          {data?.topPerformers?.slice(0, 3).map((doctor, index) => (
            <div
              key={doctor.id}
              className={`p-6 rounded-lg border-2 ${
                index === 0
                  ? 'border-yellow-400 bg-yellow-50'
                  : index === 1
                  ? 'border-gray-400 bg-gray-50'
                  : 'border-orange-400 bg-orange-50'
              }`}
            >
              <div className="text-center">
                <div className="text-4xl mb-2">
                  {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'}
                </div>
                <img
                  src={doctor.avatar || '/avatar-placeholder.png'}
                  alt={doctor.name}
                  className="w-20 h-20 rounded-full mx-auto mb-3 border-4 border-white shadow"
                />
                <div className="font-bold text-gray-800 mb-1">
                  {doctor.name}
                </div>
                <div className="text-sm text-gray-600 mb-3">
                  {doctor.specialty}
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-white rounded p-2">
                    <div className="text-gray-600">Lịch hẹn</div>
                    <div className="font-bold text-blue-600">
                      {formatNumber(doctor.appointments)}
                    </div>
                  </div>
                  <div className="bg-white rounded p-2">
                    <div className="text-gray-600">Hoàn thành</div>
                    <div className="font-bold text-green-600">
                      {doctor.completionRate}%
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Performance Comparison */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">So sánh Hiệu suất</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={data?.performanceComparison || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="appointments" name="Lịch hẹn" fill="#3b82f6" />
            <Bar dataKey="completed" name="Hoàn thành" fill="#22c55e" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Performance Metrics Radar */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Chỉ số Hiệu suất</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={data?.performanceMetrics || []}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <Radar
                name="Điểm số"
                dataKey="score"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.6}
              />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Xu hướng theo tháng</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data?.monthlyTrend || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="appointments"
                name="Lịch hẹn"
                stroke="#3b82f6"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="revenue"
                name="Doanh thu"
                stroke="#22c55e"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Performance Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold">Bảng Hiệu suất Chi tiết</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Bác sĩ
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Chuyên khoa
                </th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                  Lịch hẹn
                </th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                  Hoàn thành
                </th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                  Tỷ lệ
                </th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                  Bệnh nhân
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Doanh thu
                </th>
                <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                  Đánh giá
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data?.detailedPerformance?.map((doctor) => (
                <tr key={doctor.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <img
                        className="h-10 w-10 rounded-full"
                        src={doctor.avatar || '/avatar-placeholder.png'}
                        alt={doctor.name}
                      />
                      <div className="ml-4">
                        <div className="text-sm font-medium text-gray-900">
                          {doctor.name}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                      {doctor.specialty}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                    {formatNumber(doctor.totalAppointments)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-green-600 font-medium">
                    {formatNumber(doctor.completedAppointments)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    <span className="px-2 py-1 text-xs font-semibold bg-green-100 text-green-800 rounded">
                      {doctor.completionRate}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-900">
                    {formatNumber(doctor.uniquePatients)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-blue-600">
                    {formatCurrency(doctor.revenue)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center">
                    <div className="flex items-center justify-center">
                      <span className="text-sm font-medium text-gray-900">
                        {doctor.averageRating}
                      </span>
                      <Star className="w-4 h-4 text-yellow-500 ml-1" />
                    </div>
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