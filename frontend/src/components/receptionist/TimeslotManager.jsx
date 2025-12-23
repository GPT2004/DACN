/* eslint-disable no-unused-vars */
// src/components/receptionist/TimeSlotManager.jsx
import React, { useState } from 'react';
import { Clock, Users, Plus } from 'lucide-react';
import { mockTimeSlots } from './mockData';

const TimeSlotManager = () => {
  const [slots, setSlots] = useState(mockTimeSlots);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <Clock className="w-6 h-6" />
          Quản lý khung giờ
        </h3>
        <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 text-sm">
          <Plus className="w-4 h-4" />
          Thêm khung
        </button>
      </div>
      <div className="space-y-3">
        {slots.map((slot, idx) => (
          <div key={idx} className="flex items-center justify-between p-3 border rounded-lg">
            <div className="flex items-center gap-3">
              <Clock className="w-5 h-5 text-blue-600" />
              <span className="font-medium">{slot.time}</span>
            </div>
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-gray-600" />
              <span className="text-sm">Tối đa {slot.maxPatients} bệnh nhân</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TimeSlotManager;