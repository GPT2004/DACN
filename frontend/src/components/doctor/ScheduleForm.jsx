// src/components/doctor/ScheduleForm.jsx
import React, { useState } from 'react';

const ScheduleForm = ({ onSave, onCancel }) => {
  const [formData, setFormData] = useState({
    date: '',
    startTime: '08:00',
    endTime: '17:00',
    slotDuration: 20,
    breakStart: '12:00',
    breakEnd: '13:00',
    maxPatients: 1,
    notes: ''
  });

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-bold text-gray-900 mb-4">Tạo Lịch Làm Việc</h3>
      <form className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Ngày làm việc</label>
            <input
              type="date"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              value={formData.date}
              onChange={(e) => setFormData({ ...formData, date: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Thời lượng slot (phút)</label>
            <select
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              value={formData.slotDuration}
              onChange={(e) => setFormData({ ...formData, slotDuration: e.target.value })}
            >
              <option value="15">15 phút</option>
              <option value="20">20 phút</option>
              <option value="30">30 phút</option>
              <option value="60">60 phút</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Giờ bắt đầu</label>
            <input
              type="time"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              value={formData.startTime}
              onChange={(e) => setFormData({ ...formData, startTime: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Giờ kết thúc</label>
            <input
              type="time"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              value={formData.endTime}
              onChange={(e) => setFormData({ ...formData, endTime: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Nghỉ từ</label>
            <input
              type="time"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              value={formData.breakStart}
              onChange={(e) => setFormData({ ...formData, breakStart: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Nghỉ đến</label>
            <input
              type="time"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              value={formData.breakEnd}
              onChange={(e) => setFormData({ ...formData, breakEnd: e.target.value })}
            />
          </div>
        </div>
        <div className="flex justify-end gap-3 pt-4 border-t">
          <button type="button" onClick={onCancel} className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
            Hủy
          </button>
          <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            Lưu lịch
          </button>
        </div>
      </form>
    </div>
  );
};

export default ScheduleForm;