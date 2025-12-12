// frontend/src/components/layout/AuthLayout.jsx
import React from 'react';
import { useLocation } from 'react-router-dom';

const AuthLayout = ({ children }) => {
  const location = useLocation();
  
  const getBackgroundImage = () => {
    if (location.pathname.includes('register')) {
      return 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80';
    }
    return 'https://images.unsplash.com/photo-1505751172876-fa1923c5c528?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80';
  };

  return (
    <div className="min-h-screen flex">
      {/* Left Side - Form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-white">
        <div className="w-full max-w-md">
          {children}
        </div>
      </div>

      {/* Right Side - Image/Branding */}
      <div className="hidden lg:flex lg:flex-1 relative bg-gradient-to-br from-blue-600 to-blue-800">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-20"
          style={{ backgroundImage: `url(${getBackgroundImage()})` }}
        />
        
        <div className="relative z-10 flex flex-col justify-center p-12 text-white">
          <div className="mb-8">
            <div className="w-16 h-16 bg-white rounded-lg flex items-center justify-center mb-4">
              <span className="text-2xl font-bold text-blue-600">CMS</span>
            </div>
            <h1 className="text-4xl font-bold mb-4">
              Clinic Management System
            </h1>
            <p className="text-xl text-blue-100">
              Hệ thống quản lý phòng khám đa khoa hiện đại
            </p>
          </div>

          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-white bg-opacity-20 rounded-lg flex items-center justify-center flex-shrink-0">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-1">Quản lý hiệu quả</h3>
                <p className="text-blue-100">
                  Tối ưu hóa quy trình làm việc, tiết kiệm thời gian và chi phí
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-white bg-opacity-20 rounded-lg flex items-center justify-center flex-shrink-0">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-1">Bảo mật cao</h3>
                <p className="text-blue-100">
                  Dữ liệu bệnh nhân được mã hóa và bảo mật tuyệt đối
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-white bg-opacity-20 rounded-lg flex items-center justify-center flex-shrink-0">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-1">Tích hợp AI</h3>
                <p className="text-blue-100">
                  Hỗ trợ chẩn đoán thông minh và dự đoán nguy cơ bệnh
                </p>
              </div>
            </div>
          </div>

          <div className="mt-12 pt-8 border-t border-white border-opacity-20">
            <p className="text-sm text-blue-100">
              © 2025 Clinic Management System. Được phát triển bởi Nguyễn Thanh Tú
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthLayout;