/* eslint-disable no-unused-vars */
import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Menu, X, User, LogOut, Calendar, FileText, MessageSquare, HelpCircle, ChevronDown } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

export default function PublicHeader() {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);
  const [dropdownOpen, setDropdownOpen] = React.useState(false);
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate('/');
    setDropdownOpen(false);
  };

  const handleDashboard = () => {
    const role = user?.role?.toLowerCase() || 'patient';
    const dashboardPath = `/${role}`;
    navigate(dashboardPath);
  };

  const handleNavigate = (path) => {
    navigate(path);
    setDropdownOpen(false);
  };

  const patientMenuItems = [
    { label: '📅 Lịch hẹn', path: '/patient/appointments', icon: Calendar },
    { label: '📄 Hồ sơ', path: '/patient/medical-records', icon: FileText },
    { label: '💬 Tin nhắn', path: '/patient/messages', icon: MessageSquare },
    { label: '❓ Câu hỏi', path: '/patient/faq', icon: HelpCircle },
  ];

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="bg-white shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold">M</div>
            <span className="font-bold text-lg">Phòng Khám Tư Nhân</span>
          </Link>

          <nav className="hidden md:flex gap-6 items-center">
            <button onClick={() => scrollToSection('services')} className="text-gray-600 hover:text-emerald-600 transition">Dịch vụ</button>
            <button onClick={() => scrollToSection('doctors-section')} className="text-gray-600 hover:text-emerald-600 transition">Bác sĩ</button>
            <button onClick={() => scrollToSection('contact')} className="text-gray-600 hover:text-emerald-600 transition">Liên hệ</button>
          </nav>

          <div className="hidden md:flex gap-3 items-center">
            {isAuthenticated ? (
              <>
                <div className="relative">
                  <button
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 transition"
                  >
                    <User size={18} className="text-emerald-600" />
                    <span className="text-gray-700">{user?.full_name || 'User'}</span>
                    <ChevronDown size={16} className={`text-gray-600 transition ${dropdownOpen ? 'rotate-180' : ''}`} />
                  </button>

                  {/* Dropdown Menu */}
                  {dropdownOpen && (
                    <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-50 py-2">
                      {/* Patient Menu Items */}
                      {user?.role?.toLowerCase() === 'patient' && (
                        <>
                          {patientMenuItems.map((item) => (
                            <button
                              key={item.path}
                              onClick={() => handleNavigate(item.path)}
                              className="w-full text-left px-4 py-2 text-gray-700 hover:bg-emerald-50 hover:text-emerald-600 transition flex items-center gap-3"
                            >
                              <span className="text-lg">{item.label.split(' ')[0]}</span>
                              <span>{item.label.split(' ').slice(1).join(' ')}</span>
                            </button>
                          ))}
                          <hr className="my-2" />
                        </>
                      )}

                      {/* Doctor/Admin Menu Items */}
                      {user?.role?.toLowerCase() === 'admin' ? (
                        <button
                          onClick={() => handleNavigate('/admin')}
                          className="w-full text-left px-4 py-2 text-gray-700 hover:bg-emerald-50 hover:text-emerald-600 transition flex items-center gap-3"
                        >
                          <User size={16} />
                          <span>Quản trị Admin</span>
                        </button>
                      ) : user?.role?.toLowerCase() === 'receptionist' ? (
                        <button
                          onClick={() => handleNavigate('/receptionist')}
                          className="w-full text-left px-4 py-2 text-gray-700 hover:bg-emerald-50 hover:text-emerald-600 transition flex items-center gap-3"
                        >
                          <User size={16} />
                          <span>Trang lễ tân</span>
                        </button>
                      ) : user?.role?.toLowerCase() === 'doctor' ? (
                        <button
                          onClick={() => handleNavigate('/doctor')}
                          className="w-full text-left px-4 py-2 text-gray-700 hover:bg-emerald-50 hover:text-emerald-600 transition flex items-center gap-3"
                        >
                          <User size={16} />
                          <span>Trang bác sĩ</span>
                        </button>
                      ) : (
                        <button
                          onClick={() => handleNavigate('/patient/profile')}
                          className="w-full text-left px-4 py-2 text-gray-700 hover:bg-emerald-50 hover:text-emerald-600 transition flex items-center gap-3"
                        >
                          <User size={16} />
                          <span>Thông tin cá nhân</span>
                        </button>
                      )}

                      <button
                        onClick={handleLogout}
                        className="w-full text-left px-4 py-2 text-red-600 hover:bg-red-50 transition flex items-center gap-3"
                      >
                        <LogOut size={16} />
                        <span>Đăng xuất</span>
                      </button>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <>
                <Link to="/login" className="text-gray-600 hover:text-emerald-600">Đăng nhập</Link>
                <Link to="/register" className="bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 transition">Đăng ký</Link>
              </>
            )}
          </div>

          <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="md:hidden">
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {mobileMenuOpen && (
          <div className="md:hidden pb-4 space-y-2 border-t">
            <button onClick={() => { scrollToSection('services'); setMobileMenuOpen(false); }} className="block w-full text-left text-gray-600 hover:text-emerald-600 py-2">Dịch vụ</button>
            <button onClick={() => { scrollToSection('doctors-section'); setMobileMenuOpen(false); }} className="block w-full text-left text-gray-600 hover:text-emerald-600 py-2">Bác sĩ</button>
            <button onClick={() => { scrollToSection('contact'); setMobileMenuOpen(false); }} className="block w-full text-left text-gray-600 hover:text-emerald-600 py-2">Liên hệ</button>
            <hr className="my-2" />
            {isAuthenticated ? (
              <>
                <div className="text-gray-700 font-semibold py-2 px-2">
                  👤 {user?.full_name || 'User'}
                </div>

                {/* Mobile Patient Menu */}
                {user?.role?.toLowerCase() === 'patient' && (
                  <>
                    <button
                      onClick={() => handleNavigate('/patient/appointments')}
                      className="block w-full text-left text-gray-700 hover:text-emerald-600 py-2 px-4"
                    >
                      📅 Lịch hẹn
                    </button>
                    {/* 'Lịch sử' removed — history is shown inside Appointments page */}
                    <button
                      onClick={() => handleNavigate('/patient/medical-records')}
                      className="block w-full text-left text-gray-700 hover:text-emerald-600 py-2 px-4"
                    >
                      📄 Hồ sơ
                    </button>
                    <button
                      onClick={() => handleNavigate('/patient/messages')}
                      className="block w-full text-left text-gray-700 hover:text-emerald-600 py-2 px-4"
                    >
                      💬 Tin nhắn
                    </button>
                    <button
                      onClick={() => handleNavigate('/patient/faq')}
                      className="block w-full text-left text-gray-700 hover:text-emerald-600 py-2 px-4"
                    >
                      ❓ Câu hỏi
                    </button>
                    <hr className="my-2" />
                  </>
                )}

                <button
                  onClick={() => {
                    if (user?.role?.toLowerCase() === 'admin') {
                      handleNavigate('/admin');
                    } else if (user?.role?.toLowerCase() === 'receptionist') {
                      handleNavigate('/receptionist');
                    } else if (user?.role?.toLowerCase() === 'doctor') {
                      handleNavigate('/doctor');
                    } else {
                      handleNavigate('/patient/profile');
                    }
                  }}
                  className="block w-full text-left text-gray-700 hover:text-emerald-600 py-2 px-4"
                >
                  ⚙️ {user?.role?.toLowerCase() === 'admin' ? 'Quản trị Admin' : user?.role?.toLowerCase() === 'receptionist' ? 'Trang lễ tân' : user?.role?.toLowerCase() === 'doctor' ? 'Trang bác sĩ' : 'Thông tin cá nhân'}
                </button>
                <button
                  onClick={handleLogout}
                  className="block w-full text-left text-red-600 hover:text-red-700 py-2 px-4"
                >
                  ↪ Đăng xuất
                </button>
              </>
            ) : (
              <>
                <Link to="/login" className="block text-gray-600 hover:text-emerald-600 py-2">Đăng nhập</Link>
                <Link to="/register" className="block bg-emerald-600 text-white px-4 py-2 rounded">Đăng ký</Link>
              </>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
