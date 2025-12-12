import React, { useEffect, useState } from 'react';
import DoctorLayout from '../../components/doctor/DoctorLayout';
import { useAuth } from '../../context/AuthContext';
import { getDoctorSchedules, getDoctorByUser, createDoctorSchedule, updateDoctorSchedule, deleteDoctorSchedule } from '../../services/doctorService';

export default function SchedulePage() {
  const { user } = useAuth();
  const [schedules, setSchedules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({
    date: '',
    start_time: '08:00',
    end_time: '17:00',
    slot_duration_minutes: 20,
    capacity: 1,
    room_id: ''
  });
  const [resolvedDoctorId, setResolvedDoctorId] = useState(null);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [editingSchedule, setEditingSchedule] = useState(null);
  const [editForm, setEditForm] = useState({
    date: '',
    start_time: '08:00',
    end_time: '17:00',
    slot_duration_minutes: 20,
    capacity: 1,
    room_id: ''
  });
  const [searchDate, setSearchDate] = useState('');

  const doctorId = user?.doctors?.[0]?.id || user?.doctor_id || null;

  const loadSchedules = async (resolvedId) => {
    try {
      const startDateObj = new Date();
      startDateObj.setDate(startDateObj.getDate() - 1);
      const start = (() => {
        const y = startDateObj.getFullYear();
        const m = String(startDateObj.getMonth() + 1).padStart(2, '0');
        const d = String(startDateObj.getDate()).padStart(2, '0');
        return `${y}-${m}-${d}`;
      })();
      const endDate = new Date();
      endDate.setDate(endDate.getDate() + 14);
      const end = (() => {
        const y = endDate.getFullYear();
        const m = String(endDate.getMonth() + 1).padStart(2, '0');
        const d = String(endDate.getDate()).padStart(2, '0');
        return `${y}-${m}-${d}`;
      })();

      const res = await getDoctorSchedules(resolvedId, { startDate: start, endDate: end });
      const payload = res?.data || res || {};
      const schedulesData = payload?.data || payload;
      setSchedules(Array.isArray(schedulesData) ? schedulesData : []);
    } catch (err) {
      console.error('load schedules', err);
    }
  };

  useEffect(() => {
    (async () => {
      let resolvedDoctorId = doctorId;

      if (!resolvedDoctorId && user?.id) {
        try {
          const r = await getDoctorByUser(user.id);
          const payload = r?.data || r || {};
          const doctor = payload?.data || payload;
          resolvedDoctorId = Array.isArray(doctor) ? doctor[0]?.id : doctor?.id;
        } catch (err) {
          console.warn('could not resolve doctor by user id', err);
        }
      }

      if (!resolvedDoctorId) {
        setLoading(false);
        return;
      }

      // expose resolved id to UI
      setResolvedDoctorId(resolvedDoctorId);

      try {
        await loadSchedules(resolvedDoctorId);
      } finally {
        setLoading(false);
      }
    })();
  }, [doctorId]);

  // render time values robustly (handles '08:00:00', ISO with 1970 date, Date objects, etc.)
  const formatTime = (timeVal, dateVal) => {
    if (!timeVal) return '-';
    try {
      // plain hh:mm or hh:mm:ss
      if (typeof timeVal === 'string' && /^\d{2}:\d{2}(:\d{2})?$/.test(timeVal)) {
        return timeVal.slice(0, 5);
      }

      // ISO-like string with 1970 date (from Prisma TIME type)
      if (typeof timeVal === 'string' && timeVal.includes('1970')) {
        // Extract HH:MM from '1970-01-01T08:00:00.000Z'
        const match = timeVal.match(/T(\d{2}):(\d{2})/);
        if (match) return `${match[1]}:${match[2]}`;
      }

      // ISO-like string (contains 'T')
      if (typeof timeVal === 'string' && timeVal.includes('T')) {
        const d = new Date(timeVal);
        if (!isNaN(d)) {
          return d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
        }
      }

      // Date object
      if (timeVal instanceof Date) {
        // If it's the epoch date with time info
        if (timeVal.getUTCFullYear() === 1970) {
          const hours = String(timeVal.getUTCHours()).padStart(2, '0');
          const minutes = String(timeVal.getUTCMinutes()).padStart(2, '0');
          return `${hours}:${minutes}`;
        }
        return timeVal.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
      }

      // fallback: try to combine date and time
      if (dateVal && typeof timeVal === 'string') {
        const combined = new Date(`${dateVal}T${timeVal}`);
        if (!isNaN(combined)) {
          return combined.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
        }
      }

      // last resort: show first 5 chars
      return String(timeVal).slice(0, 5);
    } catch (err) {
      console.error('formatTime error:', err, timeVal);
      return String(timeVal).slice(0, 5);
    }
  };

  if (loading) return <DoctorLayout><div className="text-center py-8">Đang tải...</div></DoctorLayout>;

  return (
    <DoctorLayout>
      <div className="mb-4">
        {resolvedDoctorId && (
          <button onClick={() => setShowCreate(!showCreate)} className="bg-green-500 text-white px-3 py-1 rounded">{showCreate ? 'Đóng' : 'Tạo lịch'}</button>
        )}
      </div>
      {showCreate && resolvedDoctorId && (
        <div className="mb-4 p-4 bg-white rounded shadow">
          <h3 className="text-lg font-semibold mb-3">📅 Tạo Lịch Làm Việc & Timeslots</h3>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium">Ngày</label>
              <input type="date" value={form.date} onChange={e => setForm({ ...form, date: e.target.value })} className="mt-1 p-2 border rounded w-full" />
            </div>
            <div>
              <label className="block text-sm font-medium">Phòng</label>
              <div className="mt-1 p-2 border rounded w-full text-sm text-gray-600">Phòng mặc định của bác sĩ sẽ được sử dụng (nếu có)</div>
            </div>
            <div>
              <label className="block text-sm font-medium">Giờ bắt đầu</label>
              <input 
                type="time" 
                value={form.start_time.slice(0,5)} 
                onChange={e => setForm({ ...form, start_time: e.target.value })} 
                className="mt-1 p-2 border rounded w-full" 
              />
            </div>
            <div>
              <label className="block text-sm font-medium">Giờ kết thúc</label>
              <input 
                type="time" 
                value={form.end_time.slice(0,5)} 
                onChange={e => setForm({ ...form, end_time: e.target.value })} 
                className="mt-1 p-2 border rounded w-full" 
              />
            </div>
            <div>
              <label className="block text-sm font-medium">Mỗi Slot (phút)</label>
              <select 
                value={form.slot_duration_minutes} 
                onChange={e => setForm({ ...form, slot_duration_minutes: parseInt(e.target.value) })}
                className="mt-1 p-2 border rounded w-full"
              >
                <option value={15}>15 phút</option>
                <option value={20}>20 phút</option>
                <option value={30}>30 phút</option>
                <option value={45}>45 phút</option>
                <option value={60}>1 giờ (60 phút)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium">Sức chứa/Slot</label>
              <input type="number" min="1" max="10" value={form.capacity} onChange={e => setForm({ ...form, capacity: parseInt(e.target.value || 1) })} className="mt-1 p-2 border rounded w-full" />
            </div>
          </div>

          {/* Preview timeslots calculation */}
          {form.date && form.start_time && form.end_time && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
              <p className="text-sm font-medium text-blue-900">📊 Preview Timeslots:</p>
              {(() => {
                const [startH, startM] = form.start_time.split(':').map(Number);
                const [endH, endM] = form.end_time.split(':').map(Number);
                const startMin = startH * 60 + startM;
                const endMin = endH * 60 + endM;
                const duration = form.slot_duration_minutes;
                let count = 0;
                const slots = [];
                
                for (let t = startMin; t + duration <= endMin; t += duration) {
                  const slotStartH = Math.floor(t / 60);
                  const slotStartM = t % 60;
                  const slotEndH = Math.floor((t + duration) / 60);
                  const slotEndM = (t + duration) % 60;
                  const pad = n => String(n).padStart(2, '0');
                  
                  slots.push(`${pad(slotStartH)}:${pad(slotStartM)} - ${pad(slotEndH)}:${pad(slotEndM)}`);
                  count++;
                }

                return (
                  <div>
                    <p className="text-sm text-blue-800 mb-2">
                      🎯 Sẽ tạo <span className="font-bold text-lg">{count} timeslots</span> ({form.start_time} - {form.end_time}, mỗi slot {form.slot_duration_minutes} phút)
                    </p>
                    <div className="text-xs text-blue-700 max-h-24 overflow-y-auto bg-white p-2 rounded border border-blue-200">
                      {slots.map((slot, i) => (
                        <div key={i} className="py-1">
                          • {slot}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          <div className="mt-3">
            <button
              className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 font-medium"
              onClick={async () => {
                try {
                  if (!form.date) return alert('Chọn ngày');
                  setErrorMessage('');

                  // Keep date as YYYY-MM-DD string directly
                  const dateVal = form.date;

                  // Ensure times have seconds (HH:MM -> HH:MM:00)
                  const normalizeTime = (t) => {
                    if (!t) return null;
                    const timeStr = String(t).trim();
                    // Handle both "08:00" and "08:00:00" formats
                    if (timeStr.length === 5) return `${timeStr}:00`;
                    if (timeStr.length === 8) return timeStr;
                    // Try to parse and reformat
                    const parts = timeStr.split(':');
                    if (parts.length >= 2) {
                      const h = String(parts[0]).padStart(2, '0');
                      const m = String(parts[1]).padStart(2, '0');
                      return `${h}:${m}:00`;
                    }
                    return timeStr;
                  };

                  // Coerce numeric fields to expected types to avoid server errors
                  const payload = {
                    date: dateVal,
                    start_time: normalizeTime(form.start_time),
                    end_time: normalizeTime(form.end_time),
                    room_id: form.room_id ? parseInt(String(form.room_id), 10) : null,
                    slot_duration_minutes: Number(form.slot_duration_minutes) || 20,
                    capacity: Number(form.capacity) || 1
                  };

                  console.log('📤 Sending payload:', payload);
                  await createDoctorSchedule(resolvedDoctorId, payload);
                  setSuccessMessage('✅ Tạo lịch & timeslots thành công!');
                  setShowCreate(false);
                  // refresh schedules
                  await loadSchedules(resolvedDoctorId);
                } catch (err) {
                  // Show validation errors from server if present
                  const server = err.response?.data;
                  if (server) {
                    if (server.errors && Array.isArray(server.errors) && server.errors.length > 0) {
                      setErrorMessage(server.errors.map(e => `${e.field}: ${e.message}`).join('; '));
                    } else if (server.message) {
                      setErrorMessage(server.message);
                    } else {
                      setErrorMessage(err.message || 'Lỗi khi tạo lịch');
                    }
                  } else {
                    setErrorMessage(err.message || 'Lỗi khi tạo lịch');
                  }
                  console.error(err);
                }
              }}
            >
              ✅ Tạo Lịch Làm Việc
            </button>
          </div>
          {successMessage && <div className="mt-2 p-3 bg-green-100 text-green-800 rounded text-sm">{successMessage}</div>}
          {errorMessage && <div className="mt-2 p-3 bg-red-100 text-red-800 rounded text-sm">{errorMessage}</div>}
        </div>
      )}
      <div>
        <h1 className="text-2xl font-bold mb-4">Lịch làm việc</h1>
        
        {/* Search by date */}
        <div className="mb-4 p-3 bg-white rounded shadow">
          <label className="block text-sm font-medium mb-2">Tìm kiếm theo ngày</label>
          <div className="flex gap-2">
            <input 
              type="date" 
              value={searchDate} 
              onChange={(e) => setSearchDate(e.target.value)}
              className="flex-1 p-2 border rounded"
            />
            <button
              onClick={() => setSearchDate('')}
              className="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400 transition"
            >
              Xóa bộ lọc
            </button>
          </div>
        </div>

        {schedules.length === 0 ? (
          <div className="text-gray-500">Không có lịch làm việc trong khoảng thời gian đã chọn</div>
        ) : (
          <div>
            {schedules.filter(s => {
              if (!searchDate) return true;
              const scheduleDate = new Date(s.date).toISOString().split('T')[0];
              return scheduleDate === searchDate;
            }).length === 0 ? (
              <div className="text-gray-500">Không có lịch làm việc vào ngày {new Date(searchDate).toLocaleDateString('vi-VN')}</div>
            ) : (
              <div className="space-y-3">
                {schedules
                  .filter(s => {
                    if (!searchDate) return true;
                    const scheduleDate = new Date(s.date).toISOString().split('T')[0];
                    return scheduleDate === searchDate;
                  })
                  .map(s => (
              <div key={s.id} className="p-3 bg-white rounded shadow flex justify-between items-center">
                <div>
                  <div className="font-medium">{new Date(s.date).toLocaleDateString('vi-VN')}</div>
                  <div className="text-sm text-gray-500">{formatTime(s.start_time, s.date)} - {formatTime(s.end_time, s.date)} · Phòng: {s.room?.name || '-'}</div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setEditingSchedule(s.id);
                      setEditForm({
                        date: s.date,
                        start_time: formatTime(s.start_time, s.date),
                        end_time: formatTime(s.end_time, s.date),
                        slot_duration_minutes: s.slot_duration_minutes,
                        capacity: s.capacity,
                        room_id: s.room_id || ''
                      });
                    }}
                    className="bg-blue-500 text-white px-2 py-1 text-xs rounded hover:bg-blue-600"
                  >
                    Sửa
                  </button>
                  <button
                    onClick={async () => {
                      if (window.confirm('Bạn chắc chắn muốn xóa lịch này?')) {
                        try {
                          setErrorMessage('');
                          await deleteDoctorSchedule(resolvedDoctorId, s.id);
                          setSuccessMessage('Xóa lịch thành công');
                          await loadSchedules(resolvedDoctorId);
                        } catch (err) {
                          const server = err.response?.data;
                          if (server?.message) {
                            setErrorMessage(server.message);
                          } else {
                            setErrorMessage(err.message || 'Lỗi khi xóa lịch');
                          }
                          console.error(err);
                        }
                      }
                    }}
                    className="bg-red-500 text-white px-2 py-1 text-xs rounded hover:bg-red-600"
                  >
                    Xóa
                  </button>
                </div>
              </div>
            ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Edit Schedule Modal */}
      {editingSchedule && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-lg max-w-md w-full p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Sửa lịch làm việc</h2>
              <button
                onClick={() => setEditingSchedule(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm">Ngày</label>
                <input type="date" value={editForm.date} onChange={e => setEditForm({ ...editForm, date: e.target.value })} className="mt-1 p-2 border rounded w-full" />
              </div>
              <div>
                <label className="block text-sm">Phòng</label>
                <div className="mt-1 p-2 border rounded w-full text-sm text-gray-600">Phòng mặc định của bác sĩ sẽ được sử dụng (nếu có)</div>
              </div>
              <div>
                <label className="block text-sm">Giờ bắt đầu</label>
                <input 
                  type="time" 
                  value={editForm.start_time.slice(0,5)} 
                  onChange={e => setEditForm({ ...editForm, start_time: e.target.value })} 
                  className="mt-1 p-2 border rounded w-full" 
                />
              </div>
              <div>
                <label className="block text-sm">Giờ kết thúc</label>
                <input 
                  type="time" 
                  value={editForm.end_time.slice(0,5)} 
                  onChange={e => setEditForm({ ...editForm, end_time: e.target.value })} 
                  className="mt-1 p-2 border rounded w-full" 
                />
              </div>
              <div>
                <label className="block text-sm">Slot phút</label>
                <input type="number" value={editForm.slot_duration_minutes} onChange={e => setEditForm({ ...editForm, slot_duration_minutes: parseInt(e.target.value || 20) })} className="mt-1 p-2 border rounded w-full" />
              </div>
              <div>
                <label className="block text-sm">Sức chứa</label>
                <input type="number" value={editForm.capacity} onChange={e => setEditForm({ ...editForm, capacity: parseInt(e.target.value || 1) })} className="mt-1 p-2 border rounded w-full" />
              </div>
            </div>

            <div className="mt-4 flex gap-2">
              <button
                className="flex-1 bg-blue-600 text-white px-3 py-2 rounded hover:bg-blue-700"
                onClick={async () => {
                  try {
                    if (!editForm.date) return alert('Chọn ngày');
                    setErrorMessage('');

                    const dateVal = (new Date(editForm.date)).toISOString().split('T')[0];
                    const normalizeTime = (t) => {
                      if (!t) return null;
                      const s = String(t).slice(0,5);
                      return s.length === 5 ? `${s}:00` : s;
                    };

                    const payload = {
                      date: dateVal,
                      start_time: normalizeTime(editForm.start_time),
                      end_time: normalizeTime(editForm.end_time),
                      room_id: editForm.room_id ? parseInt(String(editForm.room_id), 10) : null,
                      slot_duration_minutes: Number(editForm.slot_duration_minutes) || 20,
                      capacity: Number(editForm.capacity) || 1
                    };

                    await updateDoctorSchedule(resolvedDoctorId, editingSchedule, payload);
                    setSuccessMessage('Cập nhật lịch thành công');
                    setEditingSchedule(null);
                    await loadSchedules(resolvedDoctorId);
                  } catch (err) {
                    const server = err.response?.data;
                    if (server?.errors && Array.isArray(server.errors)) {
                      setErrorMessage(server.errors.map(e => `${e.field}: ${e.message}`).join('; '));
                    } else if (server?.message) {
                      setErrorMessage(server.message);
                    } else {
                      setErrorMessage(err.message || 'Lỗi khi cập nhật lịch');
                    }
                    console.error(err);
                  }
                }}
              >
                Lưu
              </button>
              <button
                className="flex-1 bg-gray-300 text-gray-700 px-3 py-2 rounded hover:bg-gray-400"
                onClick={() => setEditingSchedule(null)}
              >
                Hủy
              </button>
            </div>
          </div>
        </div>
      )}
    </DoctorLayout>
  );
}
  // file ends here
