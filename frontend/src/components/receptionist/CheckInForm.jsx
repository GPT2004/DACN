// src/components/receptionist/CheckInForm.jsx
import React, { useState } from 'react';
import { Search, User, Phone, CheckCircle } from 'lucide-react';
import { mockAppointments } from './mockData';

const CheckInForm = ({ onCheckIn }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedAppt, setSelectedAppt] = useState(null);

  const results = mockAppointments.filter(appt =>
    appt.patientId.toString().includes(searchTerm) ||
    appt.time.includes(searchTerm)
  );

  const handleCheckIn = () => {
    if (selectedAppt && onCheckIn) {
      onCheckIn(selectedAppt);
      setSelectedAppt(null);
      setSearchTerm('');
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-bold text-gray-900 mb-4">Check-in bệnh nhân</h3>
      <div className="space-y-4">
        <div className="relative">
          <Search className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Tìm theo mã bệnh nhân hoặc giờ hẹn..."
            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        {searchTerm && (
          <div className="border border-gray-200 rounded-lg max-h-64 overflow-y-auto">
            {results.map(appt => (
              <div
                key={appt.id}
                onClick={() => setSelectedAppt(appt)}
                className={`p-3 cursor-pointer hover:bg-gray-50 border-b last:border-b-0 ${
                  selectedAppt?.id === appt.id ? 'bg-blue-50' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <User className="w-5 h-5 text-gray-600" />
                    <div>
                      <p className="font-medium">BN#{appt.patientId}</p>
                      <p className="text-sm text-gray-600">{appt.time} - {appt.reason}</p>
                    </div>
                  </div>
                  {appt.status === 'CONFIRMED' && (
                    <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">Sẵn sàng</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        {selectedAppt && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="font-medium text-blue-900">Sẵn sàng check-in:</p>
            <p className="text-sm">BN#{selectedAppt.patientId} - {selectedAppt.time}</p>
            <button
              onClick={handleCheckIn}
              className="mt-3 w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center justify-center gap-2"
            >
              <CheckCircle className="w-4 h-4" />
              Xác nhận check-in
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default CheckInForm;