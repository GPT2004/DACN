import React, {useEffect, useState} from 'react';
import AdminLayout from '../../components/admin/AdminLayout';
import StatCard from '../../components/admin/StatCard';
import { getDashboardStats } from '../../services/reportService';
import { Users, Stethoscope, Calendar, Home } from 'lucide-react';

export default function AdminDashboard(){
  const [stats, setStats] = useState({users:0, doctors:0, appointments:0, rooms:0, monthAppointments: 0, todayAppointments: 0});
  const [loading, setLoading] = useState(true);
  
  useEffect(()=>{ 
    (async ()=>{ 
      try{
        const r = await getDashboardStats();
        // Handle various response shapes
        if(r?.data) setStats(r.data);
        else if(r?.users !== undefined) setStats(r);
        else setStats({users:0, doctors:0, appointments:0, rooms:0, monthAppointments: 0, todayAppointments: 0});
      }catch(e){ 
        setStats({users:0, doctors:0, appointments:0, rooms:0, monthAppointments: 0, todayAppointments: 0});
      }finally{
        setLoading(false);
      }
    })(); 
  },[]);
  
  if(loading) return <AdminLayout><div className="text-center py-8">Đang tải...</div></AdminLayout>;
  
  return (
    <AdminLayout>
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Bảng điều khiển Quản trị Phòng khám đa khoa</h1>
          <p className="text-gray-600 mt-1">Chào mừng trở lại! Đây là tổng quan hệ thống</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard 
            icon={Users} 
            label="Tổng người dùng" 
            value={stats.users || 0} 
            color="blue"
          />
          <StatCard 
            icon={Stethoscope} 
            label="Tổng bác sĩ" 
            value={stats.doctors || 0} 
            color="green"
          />
          <StatCard 
            icon={Calendar} 
            label="Lịch hẹn khám bệnh" 
            value={stats.todayAppointments || 0} 
            color="purple"
          />
          <StatCard 
            icon={Home} 
            label="Phòng khám" 
            value={stats.rooms || 0} 
            color="orange"
          />
        </div>

        {/* Additional Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Lịch hẹn tháng này</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Tổng lịch hẹn:</span>
                <span className="text-2xl font-bold text-blue-600">{stats.appointments || 0}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition" 
                  style={{width: `${Math.min((stats.appointments || 0) / 100 * 100, 100)}%`}}
                />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Tình trạng hệ thống</h3>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="text-gray-700">Tất cả dịch vụ hoạt động bình thường</span>
              </div>
              <div className="text-sm text-gray-500 mt-3">Cập nhật lần cuối: {new Date().toLocaleString('vi-VN')}</div>
            </div>
          </div>
        </div>
      </div>
    </AdminLayout>
  );
}
