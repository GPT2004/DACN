// src/components/doctor/ScheduleCalendar.jsx
import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

const ScheduleCalendar = ({ appointments = [] }) => {
  const [currentDate] = useState(new Date());
  const [view, setView] = useState('week');
  const daysOfWeek = ['CN', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'];
  const hours = Array.from({ length: 13 }, (_, i) => i + 7);

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="p-4 border-b flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h3 className="text-lg font-bold text-gray-900">Lịch làm việc</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setView('day')}
              className={`px-3 py-1 rounded ${view === 'day' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
            >
              Ngày
            </button>
            <button
              onClick={() => setView('week')}
              className={`px-3 py-1 rounded ${view === 'week' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
            >
              Tuần
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button className="p-2 hover:bg-gray-100 rounded">
            <ChevronDown className="w-5 h-5" />
          </button>
          <span className="font-medium">
            {currentDate.toLocaleDateString('vi-VN', { month: 'long', year: 'numeric' })}
          </span>
          <button className="p-2 hover:bg-gray-100 rounded">
            <ChevronUp className="w-5 h-5" />
          </button>
        </div>
      </div>
      <div className="p-4">
        {view === 'week' && (
          <div className="grid grid-cols-8 gap-2">
            <div className="font-medium text-gray-600 text-sm">Giờ</div>
            {daysOfWeek.map(day => (
              <div key={day} className="font-medium text-center text-gray-600 text-sm">{day}</div>
            ))}
            {hours.map(hour => (
              <React.Fragment key={hour}>
                <div className="text-sm text-gray-500 py-2">{hour}:00</div>
                {daysOfWeek.map((_, idx) => (
                  <div key={`${hour}-${idx}`} className="border border-gray-200 rounded p-1 min-h-16 hover:bg-gray-50"></div>
                ))}
              </React.Fragment>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ScheduleCalendar;