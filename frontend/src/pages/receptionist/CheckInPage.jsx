/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle } from 'lucide-react';
import CheckInForm from '../../components/receptionist/CheckInForm';

export default function CheckInPage() {
  const [checkedInPatients, setCheckedInPatients] = useState([]);
  const [successMessage, setSuccessMessage] = useState('');

  const handleCheckIn = (appointmentData) => {
    try {
      // Mock check-in
      const newCheckedIn = {
        id: Math.random(),
        patientName: appointmentData.patientName || 'BN' + appointmentData.patientId,
        time: appointmentData.time,
        doctor: 'TS. Trần A',
        checkedInAt: new Date().toLocaleTimeString('vi-VN'),
        room: 'Phòng 101'
      };

      setCheckedInPatients([newCheckedIn, ...checkedInPatients]);
      setSuccessMessage(`✅ Check-in thành công cho ${newCheckedIn.patientName}`);

      setTimeout(() => {
        setSuccessMessage('');
      }, 3000);
    } catch (error) {
      console.error('Check-in error:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
          <CheckCircle className="w-8 h-8 text-green-600" />
          Check-in Bệnh Nhân
        </h1>
        <p className="text-gray-600 mt-1">Xác nhận bệnh nhân đã tới phòng khám</p>
      </div>

      {/* Success Message */}
      {successMessage && (
        <div className="p-4 bg-green-50 border border-green-200 rounded-lg flex items-start gap-3">
          <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
          <p className="text-green-900 font-medium">{successMessage}</p>
        </div>
      )}

      {/* Check-in Form */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <CheckInForm onCheckIn={handleCheckIn} />
        </div>

        {/* Tips Sidebar */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 h-fit">
          <h3 className="font-semibold text-blue-900 mb-4">📋 Quy Trình Check-in</h3>
          <ol className="space-y-3 text-sm text-blue-800">
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">1</span>
              <span>Tìm kiếm lịch hẹn của bệnh nhân</span>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">2</span>
              <span>Xác nhận thông tin bệnh nhân</span>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">3</span>
              <span>Bấm nút "Xác nhận check-in"</span>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">4</span>
              <span>Hướng dẫn bệnh nhân đến phòng khám</span>
            </li>
          </ol>

          <div className="mt-6 p-3 bg-blue-100 rounded border border-blue-300">
            <p className="text-xs text-blue-900 font-semibold">⚠️ Lưu ý:</p>
            <ul className="text-xs text-blue-800 mt-2 space-y-1">
              <li>• Chỉ check-in bệnh nhân có lịch hẹn</li>
              <li>• Kiểm tra giờ hẹn trước khi xác nhận</li>
              <li>• Cập nhật thông tin liên lạc nếu cần</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Recent Check-ins */}
      {checkedInPatients.length > 0 && (
        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b">
            <h2 className="text-lg font-semibold text-gray-900">
              Bệnh Nhân Đã Check-in Hôm Nay ({checkedInPatients.length})
            </h2>
          </div>
          <div className="divide-y max-h-96 overflow-y-auto">
            {checkedInPatients.map((patient) => (
              <div key={patient.id} className="p-4 hover:bg-gray-50 transition">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-gray-900">{patient.patientName}</p>
                    <div className="flex flex-wrap gap-4 text-sm text-gray-600 mt-1">
                      <span>🕐 Lịch hẹn: {patient.time}</span>
                      <span>👨‍⚕️ Bác sĩ: {patient.doctor}</span>
                      <span>🏥 Phòng: {patient.room}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-green-600 font-semibold">
                      ✓ Check-in {patient.checkedInAt}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info Box */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200 rounded-lg p-6">
          <div className="flex gap-4">
            <div className="flex-shrink-0">
              <CheckCircle className="w-8 h-8 text-green-600" />
            </div>
            <div>
              <h3 className="font-semibold text-green-900">Lợi Ích Của Check-in</h3>
              <ul className="text-sm text-green-800 mt-2 space-y-1">
                <li>✓ Theo dõi thời gian bệnh nhân tới</li>
                <li>✓ Tính thống kê chính xác hơn</li>
                <li>✓ Cải thiện trải nghiệm bệnh nhân</li>
                <li>✓ Quản lý hàng đợi khám hiệu quả</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-lg p-6">
          <div className="flex gap-4">
            <div className="flex-shrink-0">
              <AlertCircle className="w-8 h-8 text-blue-600" />
            </div>
            <div>
              <h3 className="font-semibold text-blue-900">Trạng Thái Lịch Hẹn</h3>
              <ul className="text-sm text-blue-800 mt-2 space-y-1">
                <li>🔵 CONFIRMED = Sẵn sàng check-in</li>
                <li>🟣 CHECKED_IN = Đã xác nhận</li>
                <li>🟢 COMPLETED = Khám xong</li>
                <li>⚪ CANCELLED = Đã hủy</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
