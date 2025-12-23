/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createAppointment } from '../../services/appointmentService';
import { patientService } from '../../services/patientService';
import TimeslotSelector from './TimeslotSelector';

export default function BookingForm({ doctorId, onSuccess, onCancel }) {
  const [step, setStep] = useState(1); // 1: doctor+date, 2: timeslot, 3: review
  const [date, setDate] = useState('');
  const [selectedTimeslotId, setSelectedTimeslotId] = useState(null);
  const [reason, setReason] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [creatingPatient, setCreatingPatient] = useState(false);
  const [newPatientData, setNewPatientData] = useState({ full_name: '', phone: '', dob: '', gender: '', address: '' });

  const navigate = useNavigate();

  const parseLocalDate = (iso) => {
    if (!iso) return new Date('');
    const parts = String(iso).split('-');
    if (parts.length !== 3) return new Date(iso);
    const y = parseInt(parts[0], 10);
    const m = parseInt(parts[1], 10);
    const d = parseInt(parts[2], 10);
    return new Date(y, m - 1, d);
  };

  useEffect(() => {
    // load patient's profiles (own + dependents)
    const loadPatients = async () => {
      try {
        const res = await patientService.getMyPatients();
        const list = res?.data || res || [];
        setPatients(Array.isArray(list) ? list : []);
        if (Array.isArray(list) && list.length === 1) {
          setSelectedPatientId(list[0].id);
        }
      } catch (err) {
        // ignore
      }
    };
    loadPatients();
  }, []);

  const handleCreatePatient = async () => {
    setLoading(true);
    try {
      const payload = { ...newPatientData };
      const res = await patientService.createDependentPatient(payload);
      const created = res?.data || res;
      const added = created || res;
      // refresh list
      const listRes = await patientService.getMyPatients();
      const list = listRes?.data || listRes || [];
      setPatients(Array.isArray(list) ? list : []);
      setSelectedPatientId(added.id || added);
      setCreatingPatient(false);
    } catch (err) {
      setError(err.response?.data?.message || err.message || 'Tạo hồ sơ thất bại');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!selectedTimeslotId || !reason) {
      setError('Vui lòng chọn khung giờ và lý do khám');
      return;
    }

    if (!selectedPatientId) {
      setError('Vui lòng chọn hồ sơ bệnh nhân hoặc tạo hồ sơ mới');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const payload = {
        timeslot_id: selectedTimeslotId,
        appointment_date: date,
        reason,
        patient_id: selectedPatientId,
      };

      const result = await createAppointment(payload);
      if (onSuccess) onSuccess(result);
    } catch (err) {
      setError(err.response?.data?.message || err.message || 'Đặt lịch thất bại');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 max-w-2xl">
      <h2 className="text-2xl font-bold mb-6">Đặt lịch khám</h2>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded">
          {error}
        </div>
      )}

      {/* Step 1: Date Selection (with 7-day quick picker) */}
      {step === 1 && (
        <div>
          <label className="block text-sm font-medium mb-2">Chọn ngày khám</label>
          <div className="mb-3 grid grid-cols-4 gap-2">
            {Array.from({ length: 7 }).map((_, idx) => {
              const d = new Date();
              d.setDate(d.getDate() + idx);
              // build local YYYY-MM-DD to avoid UTC shifts from toISOString()
              const pad = (n) => (n < 10 ? '0' + n : String(n));
              const iso = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
              const label = d.toLocaleDateString('vi-VN', { weekday: 'short', day: '2-digit', month: '2-digit' });
              // today's local ISO for comparison
              const today = (() => { const t = new Date(); return `${t.getFullYear()}-${pad(t.getMonth() + 1)}-${pad(t.getDate())}`; })();
              return (
                <button
                  key={iso}
                  onClick={() => {
                    setDate(iso);
                    setError('');
                    setStep(2);
                  }}
                  className={`p-2 text-sm rounded border ${date === iso ? 'bg-emerald-600 text-white' : 'bg-white'}`}
                >
                  <div className="font-medium">{label}</div>
                  <div className="text-xs opacity-80">{iso === today ? 'Hôm nay' : ''}</div>
                </button>
              );
            })}
          </div>

          <div className="mb-3">
            <label className="text-sm">Chọn ngày khác</label>
            <input
              type="date"
              value={date}
              onChange={(e) => {
                setDate(e.target.value);
                setError('');
                if (e.target.value) setStep(2);
              }}
              min={(() => { const t = new Date(); const pad = (n) => (n < 10 ? '0' + n : String(n)); return `${t.getFullYear()}-${pad(t.getMonth() + 1)}-${pad(t.getDate())}`; })()}
              className="w-full p-2 border rounded"
            />
          </div>
          {/* Removed explicit "Tiếp tục" button — selecting a date now shows available timeslots below */}
        </div>
      )}

      {/* Step 2: Timeslot Selection */}
      {step === 2 && (
        <div>
          <label className="block text-sm font-medium mb-2">
            Chọn khung giờ ({parseLocalDate(date).toLocaleDateString('vi-VN')})
          </label>
          <div className="mb-4">
            <TimeslotSelector
              doctorId={doctorId}
              date={date}
              selectedTimeslotId={selectedTimeslotId}
              onSelect={setSelectedTimeslotId}
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setStep(1)}
              className="flex-1 bg-gray-200 text-gray-700 py-2 rounded hover:bg-gray-300"
            >
              Quay lại
            </button>
            <button
              onClick={() => {
                if (selectedTimeslotId) setStep(3);
                else setError('Vui lòng chọn khung giờ');
              }}
              className="flex-1 bg-blue-500 text-white py-2 rounded hover:bg-blue-600"
            >
              Tiếp tục
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Review & Reason */}
      {step === 3 && (
        <div>
          <div className="mb-4 p-3 bg-gray-50 rounded">
            <p className="text-sm">
              <strong>Ngày:</strong> {parseLocalDate(date).toLocaleDateString('vi-VN')}
            </p>
          </div>

          <label className="block text-sm font-medium mb-2">Hồ sơ bệnh nhân</label>
          {patients.length === 0 ? (
            <div className="mb-3 text-sm text-gray-600">Bạn chưa có hồ sơ nào. Vui lòng tạo hồ sơ mới.</div>
          ) : (
            <div className="mb-3 grid gap-2">
              {patients.map((p) => (
                <label key={p.id} className={`p-2 border rounded cursor-pointer ${selectedPatientId === p.id ? 'bg-emerald-50 border-emerald-400' : 'bg-white'}`}>
                  <input type="radio" name="patient" value={p.id} checked={selectedPatientId === p.id} onChange={() => setSelectedPatientId(p.id)} className="mr-2" />
                  <span className="font-medium">{p.full_name || p.user?.full_name}</span>
                  <div className="text-xs text-gray-500">{p.phone || p.user?.phone} {p.dob ? `• ${new Date(p.dob).toLocaleDateString('vi-VN')}` : ''}</div>
                </label>
              ))}
            </div>
          )}

          <div className="mb-4">
            <button type="button" onClick={() => setCreatingPatient(!creatingPatient)} className="text-sm text-blue-600 underline">{creatingPatient ? 'Hủy tạo hồ sơ mới' : 'Tạo hồ sơ mới'}</button>
            {creatingPatient && (
              <div className="mt-3 p-3 border rounded bg-gray-50">
                <input className="w-full p-2 border rounded mb-2" placeholder="Họ và tên" value={newPatientData.full_name} onChange={(e) => setNewPatientData(prev => ({...prev, full_name: e.target.value}))} />
                <input className="w-full p-2 border rounded mb-2" placeholder="Số điện thoại" value={newPatientData.phone} onChange={(e) => setNewPatientData(prev => ({...prev, phone: e.target.value}))} />
                <input type="date" className="w-full p-2 border rounded mb-2" value={newPatientData.dob} onChange={(e) => setNewPatientData(prev => ({...prev, dob: e.target.value}))} />
                <select className="w-full p-2 border rounded mb-2" value={newPatientData.gender} onChange={(e) => setNewPatientData(prev => ({...prev, gender: e.target.value}))}>
                  <option value="">-- Giới tính --</option>
                  <option value="MALE">Nam</option>
                  <option value="FEMALE">Nữ</option>
                  <option value="OTHER">Khác</option>
                </select>
                <input className="w-full p-2 border rounded mb-2" placeholder="Địa chỉ" value={newPatientData.address} onChange={(e) => setNewPatientData(prev => ({...prev, address: e.target.value}))} />
                <input className="w-full p-2 border rounded mb-2" placeholder="Email" value={newPatientData.email || ''} onChange={(e) => setNewPatientData(prev => ({...prev, email: e.target.value}))} />
                <input className="w-full p-2 border rounded mb-2" placeholder="Nghề nghiệp" value={newPatientData.occupation || ''} onChange={(e) => setNewPatientData(prev => ({...prev, occupation: e.target.value}))} />
                <div className="grid grid-cols-2 gap-2 mb-2">
                  <select className="p-2 border rounded" value={newPatientData.id_type || ''} onChange={(e) => setNewPatientData(prev => ({...prev, id_type: e.target.value}))}>
                    <option value="">-- Loại giấy tờ --</option>
                    <option value="CCCD">CCCD</option>
                    <option value="CMND">CMND</option>
                    <option value="PASSPORT">Hộ chiếu</option>
                  </select>
                  <input className="p-2 border rounded" placeholder="Số giấy tờ" value={newPatientData.id_number || ''} onChange={(e) => setNewPatientData(prev => ({...prev, id_number: e.target.value}))} />
                </div>
                <select className="w-full p-2 border rounded mb-2" value={newPatientData.nationality || 'Vietnam'} onChange={(e) => setNewPatientData(prev => ({...prev, nationality: e.target.value}))}>
                  <option value="">-- Quốc gia --</option>
                  {/* small list; keep Vietnam default */}
                  <option value="Vietnam">Việt Nam</option>
                  <option value="United States">United States</option>
                  <option value="United Kingdom">United Kingdom</option>
                  <option value="China">China</option>
                  <option value="Japan">Japan</option>
                </select>
                <input className="w-full p-2 border rounded mb-2" placeholder="Dân tộc" value={newPatientData.ethnicity || ''} onChange={(e) => setNewPatientData(prev => ({...prev, ethnicity: e.target.value}))} />
                <div className="flex gap-2">
                  <button type="button" onClick={handleCreatePatient} className="flex-1 bg-emerald-600 text-white py-2 rounded">Tạo hồ sơ và chọn</button>
                  <button type="button" onClick={() => setCreatingPatient(false)} className="flex-1 bg-gray-200 text-gray-700 py-2 rounded">Hủy</button>
                </div>
              </div>
            )}
          </div>

          <label className="block text-sm font-medium mb-2">Lý do khám</label>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Mô tả triệu chứng hoặc lý do khám..."
            className="w-full p-2 border rounded mb-4"
            rows="4"
          />

          <div className="flex gap-2">
            <button
              onClick={() => setStep(2)}
              className="flex-1 bg-gray-200 text-gray-700 py-2 rounded hover:bg-gray-300"
            >
              Quay lại
            </button>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="flex-1 bg-green-500 text-white py-2 rounded hover:bg-green-600 disabled:opacity-50"
            >
              {loading ? 'Đang đặt lịch...' : 'Xác nhận đặt lịch'}
            </button>
          </div>
        </div>
      )}

      {onCancel && (
        <button
          onClick={onCancel}
          className="mt-4 w-full text-gray-500 hover:text-gray-700"
        >
          Đóng
        </button>
      )}
    </div>
  );
}
