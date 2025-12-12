import React, { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { User, LogOut, Home } from 'lucide-react';

// Import sidebar dựa trên role
import AdminSidebar from '../admin/Sidebar';
import DoctorSidebar from '../doctor/SidebarDoctor';
import ReceptionistSidebar from '../common/Sidebar'; // receptionist dùng common sidebar

export default function AdminDashboardLayout({ children, role }) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  const handleLogout = async () => {
    await logout();
    navigate('/');
  };

  // Chọn sidebar component dựa trên role
  const SidebarComponent = role === 'Admin' 
    ? AdminSidebar 
    : role === 'Doctor' 
    ? DoctorSidebar 
    : ReceptionistSidebar;

  // Tiêu đề theo role
  const getTitle = () => {
    switch(role) {
      case 'Admin': return 'Bảng điều khiển Quản trị';
      case 'Doctor': return 'Bảng điều khiển Bác sĩ';
      case 'Receptionist': return 'Bảng điều khiển Lễ tân';
      case 'Pharmacist': return 'Bảng điều khiển Dược sĩ';
      case 'LabTech': return 'Bảng điều khiển Xét nghiệm';
      default: return 'Bảng điều khiển';
    }
  };

  const getGreeting = () => {
    switch(role) {
      case 'Admin': return 'Chào mừng trở lại';
      case 'Doctor': return 'Chào';
      case 'Receptionist': return 'Chào';
      case 'Pharmacist': return 'Chào';
      case 'LabTech': return 'Chào';
      default: return 'Chào';
    }
  };

  return (
    <div className="h-screen bg-gray-50 flex overflow-hidden">
      {/* Sidebar */}
      <div
        className={`
          fixed left-0 top-0 h-screen bg-white border-r border-gray-200 
          transition-all duration-300 z-40
          ${sidebarOpen ? 'w-64' : 'w-0 overflow-hidden'}
          md:relative md:w-64 md:block
        `}
      >
        <SidebarComponent />
      </div>

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 md:hidden z-30"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Mobile Menu Toggle */}
        <div className="flex items-center px-4 py-3 bg-white border-b border-gray-200 md:hidden z-20">
          <button
            onClick={toggleSidebar}
            className="p-2 hover:bg-gray-100 rounded-md transition"
          >
            {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Topbar - Chung cho tất cả roles */}
        <header className="bg-white border-b border-gray-200 px-8 py-4 flex justify-between items-center sticky top-0 z-10">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">{getTitle()}</h1>
            <p className="text-sm text-gray-500">
              {getGreeting()}, {user?.full_name || user?.name || 'User'}
            </p>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              title="Trang chủ"
              className="flex items-center gap-2 text-gray-700 hover:text-emerald-600 px-3 py-2 rounded-md"
            >
              <Home size={18} />
              <span className="hidden md:inline">Trang chủ</span>
            </button>

            <button
              onClick={() => {
                const dashboardPath = `/${role.toLowerCase()}`;
                navigate(dashboardPath);
              }}
              className="flex items-center gap-2 text-gray-700 hover:text-emerald-600"
            >
              <User size={18} />
              <span className="hidden md:inline">{user?.full_name || user?.name}</span>
            </button>

            <button
              onClick={handleLogout}
              className="flex items-center gap-2 px-4 py-2 rounded-md bg-red-100 text-red-700 hover:bg-red-200 transition font-medium"
            >
              <LogOut size={18} />
              <span className="hidden md:inline">Đăng xuất</span>
            </button>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto">
          <div className="p-4 md:p-8">{children}</div>
        </main>
      </div>
    </div>
  );
}
