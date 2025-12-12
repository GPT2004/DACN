import React, { useEffect, useState } from 'react';
import { timeslotService } from '../../services/scheduleService';
import { X, Calendar } from 'lucide-react';

function formatDateKey(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

// Helper to format date key (YYYY-MM-DD) already defined above

export default function ScheduleModal({ doctor, onClose, onSelectTimeslot }) {
  const [weekDays, setWeekDays] = useState([]);
  const [selectedDate, setSelectedDate] = useState(null);
  // map of 'YYYY-MM-DD' => array of timeslot objects from API
  const [availableMap, setAvailableMap] = useState({});
  const [timeslots, setTimeslots] = useState([]);

  useEffect(() => {
    // build 7-day window starting today
    const start = new Date();
    const days = [];
    for (let i = 0; i < 7; i++) {
      const d = new Date(start);
      d.setDate(start.getDate() + i);
      days.push(d);
    }
    setWeekDays(days);
    setSelectedDate(days[0]);
  }, []);

  useEffect(() => {
    // When weekDays are ready, fetch timeslots for the date range and build availability map
    if (!weekDays || weekDays.length === 0 || !doctor?.id) return;

    const startKey = formatDateKey(weekDays[0]);
    const endKey = formatDateKey(weekDays[weekDays.length - 1]);

    let mounted = true;
    const loadDoctorTimeslots = async () => {
      try {
        const res = await timeslotService.getDoctorTimeslots(doctor.id, startKey, endKey);
        const payload = res?.data || res || {};
        const list = payload?.data || payload?.timeslots || (Array.isArray(payload) ? payload : []);
        // group by date string
        const map = {};
        (Array.isArray(list) ? list : (list.timeslots || [])).forEach(ts => {
          // Normalize date to YYYY-MM-DD regardless of returned shape (ISO string, Date, or plain)
          let dateObj = null;
          if (!ts) return;
          if (ts.date instanceof Date) dateObj = ts.date;
          else if (typeof ts.date === 'string') dateObj = new Date(ts.date);
          else if (ts.date && ts.date.$date) dateObj = new Date(ts.date.$date); // defensive
          if (!dateObj || isNaN(dateObj.getTime())) return;
          const dateKey = formatDateKey(dateObj);
          if (!map[dateKey]) map[dateKey] = [];
          map[dateKey].push(ts);
        });

        if (!mounted) return;
        setAvailableMap(map);

        // Ensure selectedDate is an available day; if not, pick first available
        const selKey = selectedDate ? formatDateKey(selectedDate) : null;
        if (!selKey || !map[selKey]) {
          const firstAvailable = Object.keys(map).sort()[0];
          if (firstAvailable) {
            setSelectedDate(new Date(firstAvailable));
            setTimeslots((map[firstAvailable] || []).sort((a,b) => (a.start_time || '').localeCompare(b.start_time || '')));
          } else {
            setTimeslots([]);
          }
        } else {
          setTimeslots((map[selKey] || []).sort((a,b) => (a.start_time || '').localeCompare(b.start_time || '')));
        }
      } catch (err) {
        console.warn('Could not load doctor timeslots:', err);
        setAvailableMap({});
        setTimeslots([]);
      }
    };

    loadDoctorTimeslots();
    return () => { mounted = false; };
  }, [weekDays, doctor, selectedDate]);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-auto">
        <div className="sticky top-0 bg-white p-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Calendar size={20} className="text-blue-600" />
            <div>
              <h3 className="text-lg font-semibold">Lịch làm việc - {doctor.user?.full_name || doctor.user?.name || 'Bác sĩ'}</h3>
              <p className="text-sm text-gray-500">Chọn ngày để xem khung giờ còn trống</p>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800"><X size={20} /></button>
        </div>

        <div className="p-4">
          <div className="flex gap-2 overflow-x-auto pb-3">
            {weekDays.map(d => {
              const key = formatDateKey(d);
              const isSelected = selectedDate && formatDateKey(selectedDate) === key;
              const hasAvail = !!availableMap[key];
              return (
                <button
                  key={key}
                  onClick={() => {
                    setSelectedDate(new Date(d));
                    setTimeslots(availableMap[key] || []);
                  }}
                  className={`min-w-[110px] p-3 rounded-lg border ${isSelected ? 'bg-blue-600 text-white' : 'bg-white text-gray-700'} hover:shadow ${hasAvail ? '' : 'opacity-60'} `}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium">{d.toLocaleDateString('vi-VN', { weekday: 'short' })}</div>
                      <div className="text-xs">{d.toLocaleDateString('vi-VN', { day: '2-digit', month: 'short' })}</div>
                    </div>
                    <div className="text-xs ml-2">
                      {hasAvail ? <span className="text-green-600 font-medium">Có</span> : <span className="text-gray-400">—</span>}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Khung giờ còn trống</h4>

            {timeslots.length === 0 && (
              <div className="p-4 bg-gray-50 rounded border text-sm text-gray-600">Không có khung giờ trống trong ngày này.</div>
            )}

            {timeslots.length > 0 && (
              <div className="grid grid-cols-2 gap-2">
                {timeslots.map(ts => {
                  // ts may have start_time as 'HH:MM:SS' or 'HH:MM', normalize
                  let timeLabel = '';
                  if (ts.start_time) {
                    timeLabel = ts.start_time.length > 5 ? ts.start_time.slice(0,5) : ts.start_time;
                  } else if (ts.time) {
                    timeLabel = ts.time.length > 5 ? ts.time.slice(0,5) : ts.time;
                  } else {
                    timeLabel = ts.toString();
                  }

                  return (
                    <div key={ts.id || timeLabel} className="p-3 bg-white border rounded flex items-center justify-between">
                      <div>
                        <div className="text-sm font-medium">{timeLabel}</div>
                        <div className="text-xs text-gray-500">{doctor.user?.full_name || ''}</div>
                      </div>
                      <div>
                        <button
                          className="px-3 py-1 bg-green-600 text-white rounded text-sm"
                          onClick={() => {
                            if (onSelectTimeslot) {
                              onSelectTimeslot({
                                timeslot: ts,
                                date: formatDateKey(selectedDate),
                                doctor
                              });
                            }
                          }}
                        >Chọn</button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
