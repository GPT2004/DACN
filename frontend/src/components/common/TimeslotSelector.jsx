import React, { useState, useEffect } from 'react';
import { timeslotService } from '../../services/scheduleService';

export default function TimeslotSelector({ doctorId, date, onSelect, selectedTimeslotId }) {
  const [timeslots, setTimeslots] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!doctorId || !date) return;

    const fetchTimeslots = async () => {
      setLoading(true);
      setError('');
      try {
        // eslint-disable-next-line no-console
        console.log('TimeslotSelector: fetching timeslots for doctorId=', doctorId, 'date=', date);
        const result = await timeslotService.getAvailableTimeslots(doctorId, date);
        // eslint-disable-next-line no-console
        console.log('TimeslotSelector: API response=', result);
        const slots = result?.data || result || [];
        setTimeslots(Array.isArray(slots) ? slots : []);
      } catch (err) {
        setError('Không thể tải khung giờ: ' + (err.message || 'Lỗi server'));
        setTimeslots([]);
      } finally {
        setLoading(false);
      }
    };

    fetchTimeslots();
  }, [doctorId, date]);

  const formatTime = (timeVal) => {
    if (!timeVal) return '-';
    // Backend now sends HH:MM format, so just return it directly
    if (typeof timeVal === 'string' && /^\d{2}:\d{2}/.test(timeVal)) {
      return timeVal.slice(0, 5);
    }
    // Fallback for other formats
    try {
      if (typeof timeVal === 'string' && timeVal.includes('T')) {
        return new Date(timeVal).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
      }
      return String(timeVal).slice(0, 5);
    } catch {
      return String(timeVal).slice(0, 5);
    }
  };

  if (loading) {
    return <div className="text-gray-500">Đang tải khung giờ...</div>;
  }

  if (error) {
    return <div className="text-red-500 text-sm">{error}</div>;
  }

  if (timeslots.length === 0) {
    return <div className="text-gray-500">Không có khung giờ trống</div>;
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
      {timeslots.map((slot) => (
        <button
          key={slot.id}
          onClick={() => onSelect(slot.id)}
          className={`p-2 text-sm rounded border transition ${
            selectedTimeslotId === slot.id
              ? 'bg-blue-500 text-white border-blue-600'
              : 'bg-white border-gray-300 hover:border-blue-400'
          }`}
        >
          <div className="font-medium">{formatTime(slot.start_time)}</div>
          <div className="text-xs opacity-75">
            {slot.booked_count} / {slot.max_patients}
          </div>
        </button>
      ))}
    </div>
  );
}
