import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Calendar, Clock, User, FileText } from 'lucide-react';
import DoctorAIDiagnosis from './DoctorAIDiagnosis';

const Item = ({to, icon: Icon, children}) => (
  <NavLink
    to={to}
    className={({isActive}) => `
      flex items-center gap-3 px-4 py-3 rounded-md transition
      ${isActive ? 'bg-green-100 text-green-700 font-semibold' : 'text-gray-700 hover:bg-gray-100'}
    `}
  >
    <Icon size={20} />
    <span>{children}</span>
  </NavLink>
);

export default function SidebarDoctor(){
  return (
    <aside className="w-64 bg-white border-r border-gray-200 min-h-screen flex flex-col">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-bold text-green-700">👨‍⚕️ Bác sĩ đa khoa</h2>
        <p className="text-xs text-gray-500 mt-1">Giao diện bác sĩ đa khoa</p>
      </div>

      <nav className="p-4 space-y-2 flex-1">
        <Item to="/doctor" icon={LayoutDashboard}>Dashboard</Item>
        <Item to="/doctor/appointments" icon={Calendar}>Lịch hẹn khám bệnh</Item>
        <Item to="/doctor/schedule" icon={Clock}>Lịch làm việc</Item>
        <Item to="/doctor/prescriptions" icon={FileText}>Kê toa</Item>
        <Item to="/doctor/medical-records" icon={FileText}>Hồ sơ bệnh án</Item>
        <Item to="/doctor/profile" icon={User}>Hồ sơ</Item>
        
        {/* AI Diagnosis */}
        <div className="pt-2 border-t border-gray-200">
          <DoctorAIDiagnosis />
        </div>
      </nav>

      <div className="p-4 border-t border-gray-200 text-xs text-gray-500">
        <p>v1.0.0</p>
      </div>
    </aside>
  );
}
