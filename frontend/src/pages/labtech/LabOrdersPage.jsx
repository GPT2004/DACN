import React, { useState, useEffect } from 'react';
import LabTechLayout from '../../components/layout/LabTechLayout';
import { Search, Check, Clock, FileText } from 'lucide-react';

export default function LabOrdersPage() {
  const [orders, setOrders] = useState([]);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    loadOrders();
  }, []);

  const loadOrders = () => {
    setOrders([
      {
        id: 1,
        orderNo: 'LAB001',
        patientName: 'Nguyễn Văn A',
        testType: 'Xét nghiệm máu',
        tests: ['CBC', 'Chemistry panel'],
        status: 'pending',
        createdAt: '2025-11-14 08:30',
        doctor: 'BS. Trần Quốc Tuấn'
      },
      {
        id: 2,
        orderNo: 'LAB002',
        patientName: 'Trần Thị B',
        testType: 'Siêu âm',
        tests: ['Siêu âm tim'],
        status: 'in_progress',
        createdAt: '2025-11-14 09:00',
        doctor: 'BS. Lê Minh Quang'
      },
      {
        id: 3,
        orderNo: 'LAB003',
        patientName: 'Lê Văn C',
        testType: 'Xét nghiệm nước tiểu',
        tests: ['UA'],
        status: 'completed',
        createdAt: '2025-11-14 07:30',
        doctor: 'BS. Phạm Thị Hoa'
      }
    ]);
  };

  const filteredOrders = orders.filter(o => {
    const matchesSearch = o.patientName.toLowerCase().includes(search.toLowerCase()) ||
                         o.orderNo.toLowerCase().includes(search.toLowerCase());
    const matchesFilter = filter === 'all' || o.status === filter;
    return matchesSearch && matchesFilter;
  });

  const getStatusInfo = (status) => {
    const info = {
      pending: { label: 'Chưa làm', color: 'bg-yellow-100 text-yellow-800', icon: Clock },
      in_progress: { label: 'Đang làm', color: 'bg-blue-100 text-blue-800', icon: Clock },
      completed: { label: 'Hoàn thành', color: 'bg-green-100 text-green-800', icon: Check }
    };
    return info[status];
  };

  return (
    <LabTechLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Quản lý chỉ định xét nghiệm</h1>
          <p className="text-gray-600 mt-2">Danh sách các xét nghiệm cần thực hiện</p>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search size={20} className="absolute left-3 top-3 text-gray-400" />
              <input
                type="text"
                placeholder="Tìm theo tên bệnh nhân hoặc mã chỉ định..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
              />
            </div>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            >
              <option value="all">Tất cả</option>
              <option value="pending">Chưa làm</option>
              <option value="in_progress">Đang làm</option>
              <option value="completed">Hoàn thành</option>
            </select>
          </div>
        </div>

        {/* Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Mã XN</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Bệnh nhân</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Loại XN</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Chi tiết</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Bác sĩ</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Trạng thái</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Ngày</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Hành động</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {filteredOrders.map((o) => {
                  const statusInfo = getStatusInfo(o.status);
                  return (
                    <tr key={o.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 text-sm font-medium text-blue-600">{o.orderNo}</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{o.patientName}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{o.testType}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">
                        <div className="space-y-0.5">
                          {o.tests.map((t, i) => (
                            <div key={i} className="text-xs">• {t}</div>
                          ))}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-600">{o.doctor}</td>
                      <td className="px-6 py-4">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${statusInfo.color}`}>
                          {statusInfo.label}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-600">{o.createdAt}</td>
                      <td className="px-6 py-4 text-sm">
                        <div className="flex gap-2">
                          {o.status === 'pending' && (
                            <button className="text-blue-600 hover:text-blue-800 px-3 py-1 text-xs border border-blue-600 rounded hover:bg-blue-50">
                              Bắt đầu
                            </button>
                          )}
                          {o.status === 'in_progress' && (
                            <button className="text-green-600 hover:text-green-800 px-3 py-1 text-xs border border-green-600 rounded hover:bg-green-50">
                              Hoàn thành
                            </button>
                          )}
                          {o.status === 'completed' && (
                            <button className="text-purple-600 hover:text-purple-800 px-3 py-1 text-xs border border-purple-600 rounded hover:bg-purple-50 flex items-center gap-1">
                              <FileText size={14} />
                              Kết quả
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </LabTechLayout>
  );
}
