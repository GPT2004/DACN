import React from 'react';
import { useAuth } from '../../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { LogOut, Bell } from 'lucide-react';

export default function Topbar(){
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate('/');
  };

  return (
    <header className="bg-white border-b border-gray-200 px-8 py-4 flex justify-between items-center sticky top-0 z-10">
      {/* Left: Title */}
      <div>
        <h1 className="text-2xl font-bold text-gray-800">Bảng điều khiển Quản trị Phòng khám đa khoa</h1>
        <p className="text-sm text-gray-500">Chào mừng trở lại, {user?.full_name || user?.name || 'Admin'}</p>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-6">
        {/* Notifications */}
        <button className="relative text-gray-600 hover:text-blue-600 transition">
          <Bell size={24} />
          <span className="absolute top-0 right-0 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white transform translate-x-1/2 -translate-y-1/2 bg-red-600 rounded-full">
            3
          </span>
        </button>

        {/* User Info & Avatar */}
        <div className="flex items-center gap-3 pl-6 border-l border-gray-200">
          <button
            onClick={() => navigate('/')}
            className="text-blue-600 hover:text-blue-800 font-medium text-sm"
          >
            Trang chủ
          </button>
          <div className="text-right">
            <p className="text-sm font-semibold text-gray-800">{user?.full_name || user?.name || 'Admin'}</p>
            <p className="text-xs text-gray-500">{user?.email || 'admin@clinic.com'}</p>
          </div>
          <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-lg">
            {(user?.full_name || user?.name || 'A')[0].toUpperCase()}
          </div>
        </div>

        {/* Logout Button */}
        <button
          onClick={handleLogout}
          className="ml-4 flex items-center gap-2 px-4 py-2 rounded-md bg-red-100 text-red-700 hover:bg-red-200 transition font-medium"
        >
          <LogOut size={18} />
          <span>Đăng xuất</span>
        </button>
      </div>
    </header>
  );
}
