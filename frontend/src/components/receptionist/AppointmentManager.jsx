// src/components/receptionist/AppointmentManager.jsx
import React, { useState } from 'react';
import { Calendar, Clock, User, CheckCircle, XCircle, Search, ArrowUpDown } from 'lucide-react';
import { mockAppointments, mockPatients } from './mockData';

const AppointmentManager = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filter, setFilter] = useState('ALL');
  const [sortBy, setSortBy] = useState('created'); // 'name' | 'status' | 'created'
  const [sortOrder, setSortOrder] = useState('desc'); // 'asc' | 'desc'

  const appointments = mockAppointments
    .map(appt => {
      const patient = mockPatients.find(p => p.id === appt.patientId);
      return { ...appt, patient };
    })
    .filter(appt => {
      const patientName = (appt.patient?.user?.full_name || appt.patientName || '').toLowerCase();
      const matchesSearch = patientName.includes(searchTerm.toLowerCase());
      const matchesFilter = filter === 'ALL' || appt.status === filter;
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      // Determine values for comparison
      let valA;
      let valB;
      switch (sortBy) {
        case 'name':
          valA = (a.patient?.user?.full_name || a.patientName || '').toLowerCase();
          valB = (b.patient?.user?.full_name || b.patientName || '').toLowerCase();
          break;
        case 'status':
          valA = a.status || '';
          valB = b.status || '';
          break;
        case 'created':
        default:
          // Use id as proxy for creation time (higher id = newer)
          valA = a.id;
          valB = b.id;
          break;
      }

      if (valA < valB) return sortOrder === 'asc' ? -1 : 1;
      if (valA > valB) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });

  const handleSortChange = (key) => {
    if (sortBy === key) {
      // toggle order
      setSortOrder(prev => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(key);
      // default order per key
      setSortOrder(key === 'created' ? 'desc' : 'asc');
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      CONFIRMED: 'bg-blue-100 text-blue-800',
      PENDING: 'bg-yellow-100 text-yellow-800',
      CHECKED_IN: 'bg-green-100 text-green-800',
      CANCELLED: 'bg-red-100 text-red-800'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="p-6 border-b">
        <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <Calendar className="w-6 h-6" />
          Quản lý lịch hẹn
        </h3>
      </div>
      <div className="p-6 space-y-4">
        <div className="flex flex-wrap gap-3 items-center">
          <div className="flex-1 relative">
            <Search className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Tìm bệnh nhân..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <select
            className="px-4 py-2 border border-gray-300 rounded-lg"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="ALL">Tất cả</option>
            <option value="PENDING">Chờ xác nhận</option>
            <option value="CONFIRMED">Đã xác nhận</option>
            <option value="CHECKED_IN">Đã check-in</option>
          </select>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => handleSortChange('name')}
              className={`px-3 py-2 text-sm rounded-lg border flex items-center gap-1 transition ${sortBy === 'name' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 hover:bg-gray-50'}`}
            >
              <ArrowUpDown className="w-4 h-4" /> Tên
              {sortBy === 'name' && (
                <span className="text-xs font-medium">{sortOrder === 'asc' ? '↑' : '↓'}</span>
              )}
            </button>
            <button
              type="button"
              onClick={() => handleSortChange('status')}
              className={`px-3 py-2 text-sm rounded-lg border flex items-center gap-1 transition ${sortBy === 'status' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 hover:bg-gray-50'}`}
            >
              <ArrowUpDown className="w-4 h-4" /> Trạng thái
              {sortBy === 'status' && (
                <span className="text-xs font-medium">{sortOrder === 'asc' ? '↑' : '↓'}</span>
              )}
            </button>
            <button
              type="button"
              onClick={() => handleSortChange('created')}
              className={`px-3 py-2 text-sm rounded-lg border flex items-center gap-1 transition ${sortBy === 'created' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 hover:bg-gray-50'}`}
            >
              <ArrowUpDown className="w-4 h-4" /> Thời gian tạo
              {sortBy === 'created' && (
                <span className="text-xs font-medium">{sortOrder === 'asc' ? '↑' : '↓'}</span>
              )}
            </button>
          </div>
        </div>
        <div className="space-y-3">
          {appointments.map(appt => (
            <div key={appt.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                    <User className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{appt.patient?.user?.full_name || appt.patientName}</p>
                    <p className="text-sm text-gray-600">{appt.reason}</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <p className="text-sm font-medium flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {appt.time}
                    </p>
                    <p className="text-xs text-gray-500">{appt.date}</p>
                  </div>
                  <span className={`px-3 py-1 text-xs font-medium rounded-full ${getStatusColor(appt.status)}`}>
                    {appt.status}
                  </span>
                  <div className="flex gap-1">
                    {appt.status === 'PENDING' && (
                      <>
                        <button className="p-2 text-green-600 hover:bg-green-50 rounded">
                          <CheckCircle className="w-4 h-4" />
                        </button>
                        <button className="p-2 text-red-600 hover:bg-red-50 rounded">
                          <XCircle className="w-4 h-4" />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AppointmentManager;