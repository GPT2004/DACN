/* eslint-disable no-unused-vars */
// src/components/receptionist/RoomManager.jsx
import React from 'react';
import { DoorOpen, Users, Clock } from 'lucide-react';
import { mockRooms } from './mockData';

const RoomManager = () => {
  const getStatus = (status) => {
    return status === 'AVAILABLE'
      ? { color: 'bg-green-100 text-green-800', label: 'Trống' }
      : { color: 'bg-red-100 text-red-800', label: 'Đang dùng' };
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
        <DoorOpen className="w-6 h-6" />
        Quản lý phòng khám
      </h3>
      <div className="grid grid-cols-3 gap-4">
        {mockRooms.map(room => {
          const status = getStatus(room.status);
          return (
            <div key={room.id} className="border rounded-lg p-4 text-center">
              <p className="font-semibold text-lg">{room.name}</p>
              <span className={`inline-block mt-2 px-3 py-1 text-xs font-medium rounded-full ${status.color}`}>
                {status.label}
              </span>
              {room.status === 'IN_USE' && (
                <p className="text-xs text-gray-500 mt-2 flex items-center justify-center gap-1">
                  <Clock className="w-3 h-3" />
                  15 phút
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default RoomManager;