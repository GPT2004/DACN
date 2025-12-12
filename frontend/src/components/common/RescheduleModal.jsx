import React, { useState, useEffect } from 'react';
import api from '../../config/api.config';
import TimeslotSelector from './TimeslotSelector';

export default function RescheduleModal({ appointment, onSuccess, onCancel }) {
  const [date, setDate] = useState('');
  const [selectedTimeslotId, setSelectedTimeslotId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (appointment?.appointment_date) {
      const appointmentDate = new Date(appointment.appointment_date);
      setDate(appointmentDate.toISOString().split('T')[0]);
    }
  }, [appointment]);

  const handleReschedule = async () => {
    if (!selectedTimeslotId) {
      setError('Vui lòng chọn khung giờ mới');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const response = await api.patch(
        `/appointments/${appointment.id}/reschedule`,
        { timeslotId: selectedTimeslotId }
      );

      if (onSuccess) {
        onSuccess(response.data);
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Lỗi khi dời lịch');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (timeVal) => {
    if (!timeVal) return '-';
    try {
      if (/^\d{2}:\d{2}/.test(timeVal)) {
        return timeVal.slice(0, 5);
      }
      if (typeof timeVal === 'string' && timeVal.includes('T')) {
        return new Date(timeVal).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
      }
      return String(timeVal).slice(0, 5);
    } catch {
      return String(timeVal).slice(0, 5);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full mx-4">
        <h2 className="text-2xl font-bold mb-4">Dời lịch khám</h2>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded text-sm">
            {error}
          </div>
        )}

        <div className="mb-4 p-3 bg-gray-50 rounded">
          <p className="text-sm mb-2">
            <strong>Lịch hiện tại:</strong>
          </p>
          <p className="text-sm">
            {new Date(appointment?.appointment_date).toLocaleDateString('vi-VN')}{' '}
            {formatTime(appointment?.appointment_time)} - {appointment?.doctor?.user?.full_name}
          </p>
        </div>

        <label className="block text-sm font-medium mb-2">Chọn khung giờ mới</label>
        <div className="mb-4">
          <input
            type="date"
            value={date}
            onChange={(e) => {
              setDate(e.target.value);
              setSelectedTimeslotId(null);
            }}
            min={new Date().toISOString().split('T')[0]}
            className="w-full p-2 border rounded mb-3"
          />
          {date && (
            <TimeslotSelector
              doctorId={appointment?.doctor_id}
              date={date}
              selectedTimeslotId={selectedTimeslotId}
              onSelect={setSelectedTimeslotId}
            />
          )}
        </div>

        <div className="flex gap-2">
          <button
            onClick={onCancel}
            className="flex-1 bg-gray-200 text-gray-700 py-2 rounded hover:bg-gray-300"
          >
            Hủy
          </button>
          <button
            onClick={handleReschedule}
            disabled={loading || !selectedTimeslotId}
            className="flex-1 bg-blue-500 text-white py-2 rounded hover:bg-blue-600 disabled:opacity-50"
          >
            {loading ? 'Đang dời...' : 'Xác nhận dời lịch'}
          </button>
        </div>
      </div>
    </div>
  );
}
