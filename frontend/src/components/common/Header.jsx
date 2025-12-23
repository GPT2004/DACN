/* eslint-disable no-console */
/* eslint-disable jsx-a11y/anchor-is-valid */
/* eslint-disable no-unused-vars */
// frontend/src/components/common/Header.jsx
import React, { useState } from 'react';
import { 
  Bell, 
  Menu, 
  Search, 
  Settings, 
  LogOut, 
  User,
  ChevronDown 
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { Link } from 'react-router-dom';
import APP_CONFIG from '../../config/app.config';
import { useNotifications } from '../../context/NotificationContext';
import Avatar from './Avatar';
import Badge from './Badge';

const Header = ({ onMenuClick, showMenu = true }) => {
  const { user, logout } = useAuth();
  const { unreadCount } = useNotifications();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const { notifications, fetchNotifications, fetchUnreadCount, markAsRead } = useNotifications();

  const handleLogout = async () => {
    try {
      await logout();
      // Delay để ensure state update complete
      setTimeout(() => {
        window.location.href = '/';
      }, 300);
    } catch (error) {
      console.error('Logout failed:', error);
      // Force redirect anyway
      window.location.href = '/';
    }
  };

  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
      <div className="flex items-center justify-between h-16 px-6">
        {/* Left section - Logo & Name */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center">
            <span className="text-white font-bold text-lg">M</span>
          </div>
          <span className="text-xl font-bold text-green-600">Phòng khám đa khoa</span>
        </div>

        {/* Center section - Navigation */}
        <div className="hidden md:flex items-center gap-8">
          <a href="#" className="text-gray-600 hover:text-gray-900 font-medium">Dịch vụ</a>
          <a href="#" className="text-gray-600 hover:text-gray-900 font-medium">Bác sĩ</a>
          <a href="#" className="text-gray-600 hover:text-gray-900 font-medium">Liên hệ</a>
        </div>

        {/* Right section */}
        <div className="flex items-center gap-4">
          {/* Notifications bell */}
          <div className="relative">
            <button
              onClick={async () => {
                const next = !showNotifications;
                setShowNotifications(next);
                if (next) {
                  try { await fetchNotifications({ page: 1, limit: 10 }); } catch (e) { console.error(e); }
                  try { await fetchUnreadCount(); } catch (e) { /* ignore */ }
                }
              }}
              className="p-2 rounded-full hover:bg-gray-100 transition-colors"
              aria-label="Thông báo"
            >
              <Bell size={18} className="text-gray-600" />
              {unreadCount > 0 && (
                <span className="absolute -top-0.5 -right-0.5 inline-flex items-center justify-center px-1.5 py-0.5 text-xs font-semibold leading-none text-white bg-red-600 rounded-full">
                  {unreadCount}
                </span>
              )}
            </button>

            {showNotifications && (
              <div className="absolute right-0 mt-2 w-80 bg-white border rounded shadow-lg z-50">
                <div className="p-3 border-b font-medium">Thông báo</div>
                <div className="max-h-64 overflow-auto">
                  {notifications && notifications.length > 0 ? (
                    notifications.map(n => (
                      <div key={n.id} className={`p-3 hover:bg-gray-50 cursor-pointer ${n.is_read ? '' : 'bg-white'}`} onClick={async () => {
                        try { await markAsRead(n.id); } catch(e){console.error(e);} 
                        try {
                          if (n.payload && n.payload.url) {
                            window.location.href = n.payload.url;
                            return;
                          }
                        } catch (err) { console.error(err); }
                      }}>
                        <div className="text-sm font-medium">{n.payload?.message || n.type}</div>
                        <div className="text-xs text-gray-500">{new Date(n.created_at).toLocaleString()}</div>
                      </div>
                    ))
                  ) : (
                    <div className="p-3 text-sm text-gray-600">Không có thông báo.</div>
                  )}
                </div>
                <div className="p-2 text-center border-t">
                  <a href="/notifications" className="text-sm text-blue-600">Xem tất cả</a>
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <User size={20} className="text-gray-600" />
            <span className="text-sm font-medium text-gray-900">{user?.full_name || 'Administrator'}</span>
            <ChevronDown size={16} className="text-gray-400" />
          </div>
          <button
            onClick={handleLogout}
            className="px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors text-sm font-medium"
          >
            Đăng xuất
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;