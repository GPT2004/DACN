// src/components/doctor/DoctorStats.jsx
import React from 'react';
import { Calendar, CheckCircle, Users, TrendingUp } from 'lucide-react';

const DoctorStats = () => {
  const stats = [
    { label: 'Lịch hẹn khám bệnh', value: 12, icon: Calendar, color: 'bg-blue-500', change: '+2 so với hôm qua' },
    { label: 'Đã hoàn thành', value: 8, icon: CheckCircle, color: 'bg-green-500', change: '67% tỷ lệ hoàn thành' },
    { label: 'Tổng bệnh nhân', value: 156, icon: Users, color: 'bg-purple-500', change: '+5 tuần này' },
    { label: 'Đánh giá TB', value: 4.8, icon: TrendingUp, color: 'bg-yellow-500', change: '98% hài lòng' }
  ];

  return (
    <div className="grid grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <div key={index} className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <div className={`w-12 h-12 ${stat.color} rounded-lg flex items-center justify-center`}>
              <stat.icon className="w-6 h-6 text-white" />
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-gray-900">{stat.value}</div>
            </div>
          </div>
          <div className="text-sm font-medium text-gray-600 mb-1">{stat.label}</div>
          <div className="text-xs text-gray-500">{stat.change}</div>
        </div>
      ))}
    </div>
  );
};

export default DoctorStats;