/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import LabTechLayout from '../../components/layout/LabTechLayout';
import { FlaskConical, ClipboardList, CheckCircle, AlertCircle } from 'lucide-react';

export default function LabTechDashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalOrders: 0,
    pendingTests: 0,
    completedToday: 0,
    inProgress: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setStats({
      totalOrders: 45,
      pendingTests: 12,
      completedToday: 8,
      inProgress: 5
    });
    setLoading(false);
  }, []);

  return (
    <LabTechLayout>
      <div className="space-y-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Chào mừng, {user?.full_name}
          </h1>
          <p className="text-gray-600 mt-2">Quản lý xét nghiệm và kết quả cho Phòng khám đa khoa</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Tổng chỉ định</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats.totalOrders}</p>
              </div>
              <ClipboardList className="text-blue-500" size={40} />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Chưa làm</p>
                <p className="text-3xl font-bold text-yellow-600 mt-2">{stats.pendingTests}</p>
              </div>
              <AlertCircle className="text-yellow-500" size={40} />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Đang làm</p>
                <p className="text-3xl font-bold text-purple-600 mt-2">{stats.inProgress}</p>
              </div>
              <FlaskConical className="text-purple-500" size={40} />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Hoàn thành hôm nay</p>
                <p className="text-3xl font-bold text-green-600 mt-2">{stats.completedToday}</p>
              </div>
              <CheckCircle className="text-green-500" size={40} />
            </div>
          </div>
        </div>

        {/* Recent Orders */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Chỉ định gần đây</h2>
            <div className="space-y-3">
              <div className="p-4 bg-gray-50 rounded-lg border-l-4 border-yellow-500">
                <p className="text-sm font-medium text-gray-900">Xét nghiệm máu</p>
                <p className="text-xs text-gray-600 mt-1">Bệnh nhân: Nguyễn Văn A • ID: #LAB001</p>
                <p className="text-xs text-yellow-600 mt-1">Chưa bắt đầu</p>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg border-l-4 border-purple-500">
                <p className="text-sm font-medium text-gray-900">Siêu âm</p>
                <p className="text-xs text-gray-600 mt-1">Bệnh nhân: Trần Thị B • ID: #LAB002</p>
                <p className="text-xs text-purple-600 mt-1">Đang làm</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Kết quả cần gửi</h2>
            <div className="space-y-3">
              <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                <p className="text-sm font-medium text-green-900">XN máu - Nguyễn Văn C</p>
                <p className="text-xs text-green-700 mt-1">Hoàn thành • Chờ gửi cho bác sĩ</p>
                <button className="mt-2 text-xs bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700">
                  Gửi kết quả
                </button>
              </div>
              <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                <p className="text-sm font-medium text-green-900">Siêu âm - Lê Thị D</p>
                <p className="text-xs text-green-700 mt-1">Hoàn thành • Chờ gửi cho bác sĩ</p>
                <button className="mt-2 text-xs bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700">
                  Gửi kết quả
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </LabTechLayout>
  );
}
