import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Calendar, FileText, Users, Settings } from 'lucide-react';

const Navigation = ({ items, orientation = 'horizontal' }) => {
  const location = useLocation();

  const defaultItems = [
    { key: 'home', label: 'Trang chủ', icon: Home, path: '/' },
    { key: 'appointments', label: 'Lịch hẹn', icon: Calendar, path: '/appointments' },
    { key: 'records', label: 'Hồ sơ', icon: FileText, path: '/records' },
    { key: 'patients', label: 'Bệnh nhân', icon: Users, path: '/patients' },
    { key: 'settings', label: 'Cài đặt', icon: Settings, path: '/settings' },
  ];

  const navItems = items || defaultItems;

  const isActive = (path) => location.pathname === path;

  if (orientation === 'vertical') {
    return (
      <nav className="flex flex-col gap-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <Link
              key={item.key}
              to={item.path}
              className={`
                flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all
                ${isActive(item.path)
                  ? 'bg-blue-50 text-blue-600 font-medium'
                  : 'text-gray-700 hover:bg-gray-100'
                }
              `}
            >
              {Icon && <Icon size={20} />}
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    );
  }

  return (
    <nav className="flex items-center gap-1">
      {navItems.map((item) => {
        const Icon = item.icon;
        return (
          <Link
            key={item.key}
            to={item.path}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-lg transition-all text-sm font-medium
              ${isActive(item.path)
                ? 'bg-blue-50 text-blue-600'
                : 'text-gray-700 hover:bg-gray-100'
              }
            `}
          >
            {Icon && <Icon size={18} />}
            <span className="hidden md:inline">{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
};

export default Navigation;