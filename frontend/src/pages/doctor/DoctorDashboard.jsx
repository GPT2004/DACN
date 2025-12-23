/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-console */
import React, { useEffect, useState } from 'react';
import AdminDashboardLayout from '../../components/layout/AdminDashboardLayout';
import { useAuth } from '../../context/AuthContext';
import { getDoctorStats, getDoctorAppointments, getTodayAppointments, getDoctorByUser } from '../../services/doctorService';

export default function DoctorDashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState(null);
  const [todayAppointments, setTodayAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [resolvedDoctorIdState, setResolvedDoctorIdState] = useState(null);

  const doctorId = user?.doctors?.[0]?.id || user?.doctor_id || null;

  // Robust formatter for TIME fields
  const formatTimeHHmm = (time) => {
    if (!time) return '';
    try {
      if (typeof time === 'string') {
        const m = time.match(/^(\d{1,2}):(\d{2})(?::\d{2})?/);
        if (m) return `${m[1].padStart(2, '0')}:${m[2]}`;
        const d = new Date(time);
        if (!isNaN(d.getTime())) {
          const iso = d.toISOString();
          return `${iso.substring(11, 13)}:${iso.substring(14, 16)}`;
        }
        return time;
      }
      if (time instanceof Date) {
        const iso = time.toISOString();
        return `${iso.substring(11, 13)}:${iso.substring(14, 16)}`;
      }
      if (typeof time === 'number') {
        const ms = time < 1e12 ? time * 1000 : time;
        const d = new Date(ms);
        const iso = d.toISOString();
        return `${iso.substring(11, 13)}:${iso.substring(14, 16)}`;
      }
      if (typeof time === 'object' && time?.hours !== undefined && time?.minutes !== undefined) {
        return `${String(time.hours).padStart(2, '0')}:${String(time.minutes).padStart(2, '0')}`;
      }
      return String(time);
    } catch {
      return '';
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

      try {
        // update resolved doctor id for UI
        setResolvedDoctorIdState(resolvedDoctorId);
        const today = (() => {
          const d = new Date();
          const y = d.getFullYear();
          const m = String(d.getMonth() + 1).padStart(2, '0');
          const day = String(d.getDate()).padStart(2, '0');
          return `${y}-${m}-${day}`;
        })();

        if (resolvedDoctorId) {
          const [s, a] = await Promise.all([
            getDoctorStats(resolvedDoctorId),
            getDoctorAppointments(resolvedDoctorId, {
              startDate: today,
              endDate: today,
              limit: 10,
              page: 1
            })
          ]);

          const statsPayload = s?.data || s || {};
          const statsData = statsPayload?.data || statsPayload;

          const apptsPayload = a?.data || a || {};
          const apptsData = apptsPayload?.data || apptsPayload;

          setStats(statsData || {});
          setTodayAppointments(Array.isArray(apptsData?.appointments) ? apptsData.appointments : (Array.isArray(apptsData) ? apptsData : []));
        } else {
          // fallback to "me" endpoints if backend supports them
          const [s, a] = await Promise.all([
            getDoctorStats(null).catch(() => ({})),
            getTodayAppointments({ startDate: today, endDate: today }).catch(() => ({ appointments: [] }))
          ]);

          const statsPayload = s?.data || s || {};
          const statsData = statsPayload?.data || statsPayload;

          const apptsPayload = a?.data || a || {};
          const apptsData = apptsPayload?.data || apptsPayload;

          setStats(statsData || {});
          setTodayAppointments(Array.isArray(apptsData?.appointments) ? apptsData.appointments : (Array.isArray(apptsData) ? apptsData : []));
        }
      } catch (err) {
        console.error('DoctorDashboard load error', err);
      } finally {
        setLoading(false);
      }
    })();
  }, [doctorId]);

  if (loading) return <AdminDashboardLayout role="Doctor"><div className="text-center py-8">Đang tải...</div></AdminDashboardLayout>;

  return (
    <AdminDashboardLayout role="Doctor">
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Bảng điều khiển Bác sĩ đa khoa</h1>
          <p className="text-sm text-gray-600">Tổng quan hoạt động khám chữa bệnh cho trẻ em</p>
        </div>

        { !resolvedDoctorIdState && (
          <div className="bg-yellow-50 p-4 rounded">Bạn chưa có hồ sơ bác sĩ gắn với tài khoản này.</div>
        )}

        { resolvedDoctorIdState && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-4 rounded shadow">
              <div className="text-sm text-gray-500">Tổng lịch hẹn</div>
              <div className="text-2xl font-bold">{stats?.total_appointments ?? 0}</div>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <div className="text-sm text-gray-500">Bệnh nhân</div>
              <div className="text-2xl font-bold">{stats?.total_patients ?? 0}</div>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <div className="text-sm text-gray-500">Doanh thu</div>
              <div className="text-2xl font-bold">{(stats?.total_revenue || 0).toLocaleString('vi-VN')} VND</div>
            </div>
          </div>
        )}

        <div className="bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-3">Lịch hẹn khám bệnh</h3>
          {todayAppointments.length === 0 ? (
            <div className="text-gray-500">Không có lịch hẹn khám bệnh</div>
          ) : (
            <ul className="space-y-2">
              {todayAppointments.map(a => (
                <li key={a.id} className="p-3 border rounded flex justify-between items-center">
                  <div>
                    <div className="font-medium">{a.patient?.user?.full_name || 'Bệnh nhân'}</div>
                    <div className="text-sm text-gray-500">
                      {a.appointment_date ? new Date(a.appointment_date).toLocaleDateString('vi-VN') + ' ' : ''}
                      {formatTimeHHmm(a.appointment_time || a.timeslot?.start_time)} — {a.status}
                    </div>
                  </div>
                  <div className="text-sm text-gray-500">Phòng: {a.timeslot?.room_id || '-'}</div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </AdminDashboardLayout>
  );
}