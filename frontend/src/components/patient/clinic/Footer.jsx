import React from 'react';

export default function Footer(){
  return (
    <footer className="bg-gray-800 text-gray-200 mt-12">
      <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-start gap-6">
        <div>
          <div className="font-bold text-lg">Phòng khám đa khoa</div>
          <div className="text-sm mt-2">© {new Date().getFullYear()} Phòng khám đa khoa. Bảo lưu mọi quyền.</div>
        </div>
        <div className="text-sm">
          <div>Địa chỉ: 123 Đường Sức Khỏe, Quận Y</div>
          <div>Hotline: (024) 1234 5678</div>
        </div>
      </div>
    </footer>
  );
}
