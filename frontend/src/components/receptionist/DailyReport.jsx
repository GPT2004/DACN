// src/components/receptionist/DailyReport.jsx
import React from 'react';
import { TrendingUp, Users, DollarSign, Calendar } from 'lucide-react';

const DailyReport = () => {
  const stats = [
    { label: 'Tổng lượt khám', value: 28, icon: Users, color: 'bg-blue-500', change: '+5 so với hôm qua' },
    { label: 'Doanh thu', value: '18,450,000 ₫', icon: DollarSign, color: 'bg-green-500', change: 'Tăng 12%' },
    { label: 'Hóa đơn', value: 24, icon: Calendar, color: 'bg-purple-500', change: '95% hoàn thành' },
    { label: 'Check-in', value: 26, icon: TrendingUp, color: 'bg-yellow-500', change: '93% đúng giờ' },
  ];

  return (
    <div className="grid grid-cols-4 gap-4">
      {stats.map((stat, i) => (
        <div key={i} className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-3">
            <div className={`w-12 h-12 ${stat.color} rounded-lg flex items-center justify-center`}>
              <stat.icon className="w-6 h-6 text-white" />
            </div>
          </div>
          <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
          <p className="text-sm text-gray-600 mt-1">{stat.label}</p>
          <p className="text-xs text-gray-500 mt-1">{stat.change}</p>
        </div>
      ))}
    </div>
  );
};

export default DailyReport;