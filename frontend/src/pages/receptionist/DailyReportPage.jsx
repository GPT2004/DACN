/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Users, DollarSign, Calendar, Download, Filter } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export default function DailyReportPage() {
  const getLocalDate = (d = new Date()) => {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  };
  const [selectedDate, setSelectedDate] = useState(getLocalDate());
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Mock data generator
  const generateReportData = (date) => {
    const appointmentsByHour = [
      { hour: '08:00', appointments: 3, completed: 2 },
      { hour: '09:00', appointments: 5, completed: 5 },
      { hour: '10:00', appointments: 4, completed: 4 },
      { hour: '11:00', appointments: 2, completed: 2 },
      { hour: '13:00', appointments: 4, completed: 3 },
      { hour: '14:00', appointments: 6, completed: 5 },
      { hour: '15:00', appointments: 3, completed: 3 },
      { hour: '16:00', appointments: 2, completed: 1 },
    ];

    const revenueByService = [
      { name: 'Khám tổng quát', value: 1500000, percentage: 35 },
      { name: 'Xét nghiệm', value: 1200000, percentage: 28 },
      { name: 'Siêu âm', value: 900000, percentage: 21 },
      { name: 'Khác', value: 600000, percentage: 16 },
    ];

    const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444'];

    return {
      date: date,
      summary: {
        totalAppointments: 29,
        completedAppointments: 25,
        pendingAppointments: 4,
        totalRevenue: 4200000,
        averagePerAppointment: 168000,
        unpaidAmount: 500000,
        paidAmount: 3700000,
        totalPatients: 24,
        newPatients: 3,
        returningPatients: 21,
      },
      appointmentsByHour,
      revenueByService,
      colors: COLORS,
      payments: [
        { id: 1, method: 'Tiền mặt', amount: 2100000, count: 12 },
        { id: 2, method: 'Thẻ', amount: 1200000, count: 8 },
        { id: 3, method: 'QR Code', amount: 400000, count: 5 },
      ],
      staffPerformance: [
        { name: 'TS. Trần A', appointments: 8, revenue: 1300000, completionRate: 100 },
        { name: 'TS. Nguyễn B', appointments: 7, revenue: 1100000, completionRate: 100 },
        { name: 'TS. Phạm C', appointments: 9, revenue: 1400000, completionRate: 88 },
        { name: 'TS. Lê D', appointments: 5, revenue: 800000, completionRate: 100 },
      ]
    };
  };

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setReportData(generateReportData(selectedDate));
      setLoading(false);
    }, 500);
  }, [selectedDate]);

  const handleExport = () => {
    if (!reportData) return;
    const content = generateCSV(reportData);
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(content));
    element.setAttribute('download', `report_${reportData.date}.csv`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const generateCSV = (data) => {
    let csv = `Báo Cáo Hàng Ngày\n`;
    csv += `Ngày: ${new Date(data.date).toLocaleDateString('vi-VN')}\n\n`;
    
    csv += `TỔNG HỢP\n`;
    csv += `Tổng lịch hẹn,${data.summary.totalAppointments}\n`;
    csv += `Hoàn thành,${data.summary.completedAppointments}\n`;
    csv += `Chờ xử lý,${data.summary.pendingAppointments}\n`;
    csv += `Tổng doanh thu,${data.summary.totalRevenue}\n`;
    csv += `Đã thanh toán,${data.summary.paidAmount}\n`;
    csv += `Chưa thanh toán,${data.summary.unpaidAmount}\n`;
    csv += `Tổng bệnh nhân,${data.summary.totalPatients}\n`;
    csv += `BN mới,${data.summary.newPatients}\n\n`;

    return csv;
  };

  if (loading) {
    return (
      <div className="p-8 text-center">
        <p className="text-gray-600">Đang tải báo cáo...</p>
      </div>
    );
  }

  if (!reportData) {
    return (
      <div className="p-8 text-center">
        <p className="text-gray-600">Không có dữ liệu</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
            <BarChart3 className="w-8 h-8 text-blue-600" />
            Báo Cáo Hàng Ngày
          </h1>
          <p className="text-gray-600 mt-1">Thống kê hoạt động phòng khám</p>
        </div>
        <div className="flex gap-3">
          <div className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-gray-600" />
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            onClick={handleExport}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
          >
            <Download className="w-5 h-5" />
            Xuất CSV
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm">Tổng Lịch Hẹn</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">{reportData.summary.totalAppointments}</p>
              <p className="text-xs text-gray-500 mt-1">
                ✓ {reportData.summary.completedAppointments} hoàn thành
              </p>
            </div>
            <Calendar className="text-blue-500 w-12 h-12 opacity-20" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm">Tổng Bệnh Nhân</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">{reportData.summary.totalPatients}</p>
              <p className="text-xs text-green-600 mt-1">
                + {reportData.summary.newPatients} bệnh nhân mới
              </p>
            </div>
            <Users className="text-green-500 w-12 h-12 opacity-20" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm">Tổng Doanh Thu</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {(reportData.summary.totalRevenue / 1000000).toFixed(1)}M
              </p>
              <p className="text-xs text-green-600 mt-1">
                ✓ {(reportData.summary.paidAmount / 1000000).toFixed(1)}M đã thu
              </p>
            </div>
            <DollarSign className="text-orange-500 w-12 h-12 opacity-20" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-600 text-sm">Chưa Thanh Toán</p>
              <p className="text-3xl font-bold text-red-600 mt-2">
                {(reportData.summary.unpaidAmount / 1000000).toFixed(1)}M
              </p>
              <p className="text-xs text-gray-600 mt-1">
                Trung bình / lịch: {reportData.summary.averagePerAppointment.toLocaleString('vi-VN')}₫
              </p>
            </div>
            <TrendingUp className="text-red-500 w-12 h-12 opacity-20" />
          </div>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Appointments by Hour */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Lịch Hẹn Theo Giờ</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={reportData.appointmentsByHour}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="appointments" fill="#3B82F6" name="Tất cả" />
              <Bar dataKey="completed" fill="#10B981" name="Hoàn thành" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Revenue by Service */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Doanh Thu Theo Dịch Vụ</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={reportData.revenueByService}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.name}: ${entry.percentage}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {reportData.revenueByService.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={reportData.colors[index % reportData.colors.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => value.toLocaleString('vi-VN')} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Payment Methods */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Phương Thức Thanh Toán</h2>
          <div className="space-y-3">
            {reportData.payments.map((payment) => (
              <div key={payment.id} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <div>
                  <p className="font-medium text-gray-900">{payment.method}</p>
                  <p className="text-sm text-gray-600">{payment.count} giao dịch</p>
                </div>
                <p className="font-bold text-gray-900">{(payment.amount / 1000000).toFixed(2)}M₫</p>
              </div>
            ))}
          </div>
        </div>

        {/* Staff Performance */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Hiệu Suất Bác Sĩ</h2>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {reportData.staffPerformance.map((staff, idx) => (
              <div key={idx} className="p-3 bg-gray-50 rounded border-l-4 border-blue-600">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-gray-900">{staff.name}</p>
                    <p className="text-xs text-gray-600">{staff.appointments} lịch • {(staff.revenue / 1000000).toFixed(2)}M₫</p>
                  </div>
                  <div className="text-right">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      staff.completionRate === 100
                        ? 'bg-green-100 text-green-800'
                        : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {staff.completionRate}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Revenue by Service Details Table */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Chi Tiết Doanh Thu Dịch Vụ</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Dịch Vụ</th>
                <th className="px-6 py-3 text-right text-sm font-semibold text-gray-900">Doanh Thu</th>
                <th className="px-6 py-3 text-right text-sm font-semibold text-gray-900">Tỷ Lệ</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {reportData.revenueByService.map((service, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-sm font-medium text-gray-900">{service.name}</td>
                  <td className="px-6 py-4 text-sm text-right font-semibold text-gray-900">
                    {service.value.toLocaleString('vi-VN')} ₫
                  </td>
                  <td className="px-6 py-4 text-sm text-right">
                    <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-semibold">
                      {service.percentage}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot className="bg-gray-50 font-semibold">
              <tr>
                <td className="px-6 py-3 text-sm text-gray-900">Tổng Cộng</td>
                <td className="px-6 py-3 text-sm text-right text-gray-900">
                  {reportData.summary.totalRevenue.toLocaleString('vi-VN')} ₫
                </td>
                <td className="px-6 py-3 text-sm text-right text-gray-900">100%</td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  );
}
