import React from 'react';
import { Users, Calendar, DollarSign, FileText, Stethoscope } from 'lucide-react';

export default function ReceptionistSidebar({ stats = {}, counts = {}, pending = 0, activeTab, setActiveTab }) {
  return (
    <aside className="w-64 bg-white rounded-lg shadow p-4 sticky top-6 h-[calc(100vh-48px)] overflow-auto">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Tổng quan lễ tân Phòng khám đa khoa</h3>
        <p className="text-sm text-gray-500 mt-1">Tóm tắt nhanh các mục quan trọng</p>
      </div>

      <div className="space-y-3 mb-4">
        <button onClick={() => setActiveTab('overview')} className={`w-full text-left p-3 rounded-lg ${activeTab === 'overview' ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'} hover:bg-gray-100`}> 
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Tổng bệnh nhân</p>
              <p className="text-xl font-bold text-gray-900">{stats.totalPatients ?? counts.patients ?? 0}</p>
            </div>
            <Users className="text-blue-400" />
          </div>
        </button>

        <button onClick={() => setActiveTab('appointments')} className={`w-full text-left p-3 rounded-lg ${activeTab === 'appointments' ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'} hover:bg-gray-100`}> 
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Lịch hẹn hôm nay</p>
              <p className="text-xl font-bold text-gray-900">{stats.todayAppointments ?? counts.todayAppointments ?? 0}</p>
            </div>
            <Calendar className="text-purple-400" />
          </div>
        </button>

        <button onClick={() => setActiveTab('invoices')} className={`w-full text-left p-3 rounded-lg ${activeTab === 'invoices' ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'} hover:bg-gray-100`}> 
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Hóa đơn (chờ)</p>
              <p className="text-xl font-bold text-gray-900">{stats.pendingInvoices ?? counts.pendingInvoices ?? 0}</p>
            </div>
            <DollarSign className="text-orange-400" />
          </div>
        </button>

        <button onClick={() => setActiveTab('doctors')} className={`w-full text-left p-3 rounded-lg ${activeTab === 'doctors' ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'} hover:bg-gray-100`}> 
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Bác sĩ</p>
              <p className="text-xl font-bold text-gray-900">{counts.doctors ?? 0}</p>
            </div>
            <Stethoscope className="text-green-400" />
          </div>
        </button>
      </div>

      <div className="mt-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Đơn chờ</h4>
        <div className="bg-white border rounded p-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Đơn chờ lập hóa đơn</p>
              <p className="text-lg font-semibold text-gray-900">{pending ?? 0}</p>
            </div>
            <FileText className="text-indigo-400" />
          </div>
          <div className="mt-3">
            <button onClick={() => setActiveTab('invoices')} className="w-full px-3 py-2 bg-indigo-600 text-white rounded">Quản lý hóa đơn</button>
          </div>
        </div>
      </div>
    </aside>
  );
}
