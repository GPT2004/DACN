import React, {useEffect, useState} from 'react';
import AdminLayout from '../../components/admin/AdminLayout';
import StatCard from '../../components/admin/StatCard';
import { getDashboardStats } from '../../services/reportService';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, Users, Calendar, DollarSign } from 'lucide-react';

export default function ReportsPage(){
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(()=>{ 
    (async ()=>{ 
      try{ 
        const r = await getDashboardStats(); 
        const data = r?.data || r || {};

        // Normalize backend response keys to what the frontend expects
        const normalized = {
          totalPatients: data.totalPatients ?? data.users ?? 0,
          totalDoctors: data.totalDoctors ?? data.doctors ?? 0,
          monthAppointments: data.monthAppointments ?? data.appointments ?? 0,
          todayAppointments: data.todayAppointments ?? 0,
          todayRevenue: data.todayRevenue ?? 0,
          monthRevenue: data.monthRevenue ?? 0,
          appointmentsByStatus: data.appointmentsByStatus ?? data.appointmentsByStatus ?? {},
        };

        setSummary(normalized);
      }catch(e){ 
        setSummary(null);
      }finally{
        setLoading(false);
      }
    })(); 
  },[]);
  
  if(loading) return <AdminLayout><div className="text-center py-8">Đang tải...</div></AdminLayout>;
  
  // Sample data for charts (in real app, would come from API)
  const appointmentData = [
    { name: 'Thứ 2', lịch: 12, confirmed: 10 },
    { name: 'Thứ 3', lịch: 15, confirmed: 12 },
    { name: 'Thứ 4', lịch: 10, confirmed: 8 },
    { name: 'Thứ 5', lịch: 18, confirmed: 15 },
    { name: 'Thứ 6', lịch: 20, confirmed: 18 },
    { name: 'Thứ 7', lịch: 8, confirmed: 7 },
  ];

  const revenueData = [
    { name: 'Tuần 1', revenue: 2400 },
    { name: 'Tuần 2', revenue: 2210 },
    { name: 'Tuần 3', revenue: 2290 },
    { name: 'Tuần 4', revenue: 2000 },
  ];

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'];

  const statusData = [
    { name: 'Đã xác nhận', value: (summary?.appointmentsByStatus?.CONFIRMED || 0) },
    { name: 'Chờ xác nhận', value: (summary?.appointmentsByStatus?.PENDING || 0) },
    { name: 'Đã hoàn thành', value: (summary?.appointmentsByStatus?.COMPLETED || 0) },
    { name: 'Hủy', value: (summary?.appointmentsByStatus?.CANCELLED || 0) },
  ].filter(item => item.value > 0);

  return (
    <AdminLayout>
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Báo cáo & Thống kê</h1>
          <p className="text-gray-600 mt-1">Tổng quan hoạt động của phòng khám</p>
        </div>

        {/* Key Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard 
            icon={Users} 
            label="Tổng người dùng" 
            value={summary?.totalPatients || 0} 
            color="blue"
          />
          <StatCard 
            icon={TrendingUp} 
            label="Tổng bác sĩ" 
            value={summary?.totalDoctors || 0} 
            color="green"
          />
          <StatCard 
            icon={Calendar} 
            label="Lịch hẹn tháng này" 
            value={summary?.monthAppointments || 0} 
            color="purple"
          />
          <StatCard 
            icon={DollarSign} 
            label="Doanh thu tháng" 
            value={`${(summary?.monthRevenue || 0).toLocaleString('vi-VN')} VND`} 
            color="orange"
          />
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Appointment Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Lịch hẹn trong tuần</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={appointmentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="lịch" fill="#3b82f6" name="Tổng lịch hẹn" />
                <Bar dataKey="confirmed" fill="#10b981" name="Đã xác nhận" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Status Pie Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Trạng thái lịch hẹn</h3>
            {statusData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={statusData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {statusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-80 flex items-center justify-center text-gray-500">Không có dữ liệu</div>
            )}
          </div>

          {/* Revenue Chart */}
          <div className="bg-white rounded-lg shadow p-6 lg:col-span-2">
            <h3 className="text-lg font-semibold mb-4">Doanh thu theo tuần</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={revenueData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="revenue" 
                  stroke="#f59e0b" 
                  name="Doanh thu (VND)"
                  dot={{ fill: '#f59e0b', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h4 className="font-semibold text-gray-800 mb-2">Doanh thu hôm nay</h4>
            <p className="text-3xl font-bold text-blue-600">{(summary?.todayRevenue || 0).toLocaleString('vi-VN')} VND</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h4 className="font-semibold text-gray-800 mb-2">Lịch hẹn khám bệnh</h4>
            <p className="text-3xl font-bold text-green-600">{summary?.todayAppointments || 0}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-6">
            <h4 className="font-semibold text-gray-800 mb-2">Người dùng hoạt động</h4>
            <p className="text-3xl font-bold text-purple-600">{summary?.totalPatients || 0}</p>
          </div>
        </div>
      </div>
    </AdminLayout>
  );
}
