/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createAppointment } from '../../../services/appointmentService';
import { getAllDoctorsPublic, getSpecialties } from '../../../services/doctorService';
import TimeslotSelector from '../../common/TimeslotSelector';
import { useAuth } from '../../../context/AuthContext';

export default function BookingForm({ defaultDoctorId = null }) {
  const navigate = useNavigate();
  const { isAuthenticated, user } = useAuth();
  
  const [doctorId, setDoctorId] = useState(defaultDoctorId);
  const [date, setDate] = useState('');
  const [timeslotId, setTimeslotId] = useState(null);
  const [reason, setReason] = useState('');
  const [doctors, setDoctors] = useState([]);
  const [specialties, setSpecialties] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showLoginPrompt, setShowLoginPrompt] = useState(false);

  useEffect(() => {
    let m = true;
    const load = async () => {
      try {
        const dres = await getAllDoctorsPublic({ limit: 50 });
        const sres = await getSpecialties();
        if (!m) return;
        setDoctors(dres.data.doctors || dres.data || []);
        const specs = Array.isArray(sres.data) ? sres.data : (sres.data?.specialties || []);
        setSpecialties(specs);
      } catch (err) {
        console.error('BookingForm load error', err);
      }
    };
    load();
    return () => { m = false; };
  }, []);

  useEffect(() => {
    if (defaultDoctorId) setDoctorId(Number(defaultDoctorId));
  }, [defaultDoctorId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Kiểm tra đăng nhập trước
    if (!isAuthenticated) {
      setShowLoginPrompt(true);
      return;
    }
    
    if (!timeslotId) {
      alert('Vui lòng chọn khung giờ khám');
      return;
    }
    
    try {
      setLoading(true);
      const payload = { 
        timeslot_id: timeslotId,
        reason 
      };
      
      const result = await createAppointment(payload);
      alert('Đặt lịch thành công! Chúng tôi sẽ liên hệ xác nhận.');
      
      // Reset form
      setDate(''); 
      setTimeslotId(null);
      setReason('');
    } catch (error) {
      console.error('Create appointment error', error);
      const errorMsg = error.response?.data?.message || error.message || 'Đặt lịch thất bại. Vui lòng thử lại.';
      alert(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleLoginRedirect = () => {
    const currentPath = window.location.pathname;
    navigate(`/login?returnUrl=${encodeURIComponent(currentPath)}`);
  };

  return (
    <div className="space-y-4">
      {/* Thông báo yêu cầu đăng nhập */}
      {!isAuthenticated && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                Bạn cần <button onClick={handleLoginRedirect} className="font-semibold underline hover:text-yellow-800">đăng nhập</button> để đặt lịch khám
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Modal xác nhận đăng nhập */}
      {showLoginPrompt && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Yêu cầu đăng nhập</h3>
            <p className="text-gray-600 mb-6">
              Bạn cần đăng nhập để đặt lịch khám. Vui lòng đăng nhập hoặc tạo tài khoản mới.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowLoginPrompt(false)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Hủy
              </button>
              <button
                onClick={handleLoginRedirect}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Đăng nhập
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Form đặt lịch */}
      <form onSubmit={handleSubmit} className="space-y-3">
        {isAuthenticated && user && (
          <div className="bg-blue-50 border-l-4 border-blue-400 p-3 rounded">
            <p className="text-sm text-blue-700">
              <span className="font-semibold">Xin chào, {user.full_name}!</span> Vui lòng chọn thông tin đặt lịch khám.
            </p>
          </div>
        )}

        <select 
          value={doctorId || ''} 
          onChange={e => {
            const v = e.target.value;
            setDoctorId(v ? Number(v) : '');
            setTimeslotId(null);
          }} 
          className="w-full border rounded px-3 py-2"
          required
          disabled={!isAuthenticated}
        >
          <option value="">-- Chọn bác sĩ --</option>
          {doctors.map(d => (
            <option key={d.id || d.userId} value={d.id || d.userId}>
              {d.name || d.fullName || d.user?.name} - {d.specialty || d.specialties?.[0]}
            </option>
          ))}
        </select>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Chọn ngày khám
          </label>
          <input 
            type="date" 
            value={date} 
            onChange={e => {
              setDate(e.target.value);
              setTimeslotId(null);
            }}
            min={new Date().toISOString().split('T')[0]}
            className="w-full border rounded px-3 py-2" 
            required
            disabled={!isAuthenticated}
          />
        </div>

        {doctorId && date && isAuthenticated && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chọn khung giờ
            </label>
            <TimeslotSelector
              doctorId={doctorId}
              date={date}
              selectedTimeslotId={timeslotId}
              onSelect={setTimeslotId}
            />
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Lý do khám / Triệu chứng
          </label>
          <textarea 
            value={reason} 
            onChange={e => setReason(e.target.value)} 
            placeholder="Mô tả triệu chứng hoặc lý do khám bệnh" 
            className="w-full border rounded px-3 py-2" 
            rows={3}
            required
            disabled={!isAuthenticated}
          />
        </div>

        <button 
          type="submit" 
          disabled={loading || !timeslotId || !isAuthenticated} 
          className="w-full bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {!isAuthenticated ? 'Vui lòng đăng nhập' : loading ? 'Đang gửi...' : 'Xác nhận đặt lịch'}
        </button>
      </form>
    </div>
  );
}
