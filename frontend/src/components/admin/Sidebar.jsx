import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Users, Stethoscope, BarChart3, FileText, Archive, MapPin, Heart } from 'lucide-react';

const Item = ({to, icon: Icon, children, onClick}) => {
  if (onClick) {
    return (
      <button
        onClick={onClick}
        className="w-full flex items-center gap-3 px-4 py-3 rounded-md transition text-gray-700 hover:bg-gray-100 text-left"
      >
        <Icon size={20} />
        <span>{children}</span>
      </button>
    );
  }
  
  return (
    <NavLink
      to={to}
      className={({isActive}) => `
        flex items-center gap-3 px-4 py-3 rounded-md transition
        ${isActive ? 'bg-blue-100 text-blue-700 font-semibold' : 'text-gray-700 hover:bg-gray-100'}
      `}
    >
      <Icon size={20} />
      <span>{children}</span>
    </NavLink>
  );
};

export default function Sidebar(){
  return (
    <aside className="w-64 bg-white border-r border-gray-200 min-h-screen flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-bold text-blue-700">
          🏥 Phòng khám đa khoa Admin
        </h2>
        <p className="text-xs text-gray-500 mt-1">Phòng khám đa khoa</p>
      </div>

      {/* Navigation */}
      <nav className="p-4 space-y-2 flex-1">
        <Item to="/admin" icon={LayoutDashboard}>Dashboard</Item>
        <Item to="/admin/users" icon={Users}>Người dùng</Item>
        <Item to="/admin/doctors" icon={Stethoscope}>Bác sĩ</Item>
        <Item to="/admin/rooms" icon={MapPin}>Phòng khám</Item>
        <Item to="/admin/specialties" icon={Heart}>Chuyên khoa</Item>
        {/* Admin tools */}
        <Item to="/admin/audit-logs" icon={FileText}>Audit Logs</Item>
        <Item to="/admin/recycle-bin" icon={Archive}>Thùng rác</Item>
        <Item to="/admin/reports" icon={BarChart3}>Báo cáo</Item>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 text-xs text-gray-500">
        <p>v1.0.0</p>
      </div>
    </aside>
  );
}
