import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { medicalRecordService } from '../../services/medicalRecordService';
import { patientService } from '../../services/patientService';
import { prescriptionService } from '../../services/prescriptionService';

export default function MedicalRecordsPage() {
  const [records, setRecords] = useState([]);
  const [filteredRecords, setFilteredRecords] = useState([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const searchTimer = useRef(null);
  const [showModal, setShowModal] = useState(false);
  const [showDetail, setShowDetail] = useState(false);
  const [sending, setSending] = useState(false);
  const [redirectAfterCreate, setRedirectAfterCreate] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({ patientName: '', diagnosis: '', notes: '' });
  const [patientInfo, setPatientInfo] = useState(null);
  const [detailRecord, setDetailRecord] = useState(null);
  const navigate = useNavigate();
  const { user } = useAuth();

  // Helper: highlight the first match of `q` inside `text` (case-insensitive)
  const renderHighlighted = (text = '', q = '') => {
    if (!q) return text;
    const txt = (text || '').toString();
    const qLow = q.toString().toLowerCase();
    const txtLow = txt.toLowerCase();
    const idx = txtLow.indexOf(qLow);
    if (idx === -1) return txt;
    const before = txt.slice(0, idx);
    const match = txt.slice(idx, idx + qLow.length);
    const after = txt.slice(idx + qLow.length);
    return (
      <span>
        {before}
        <mark className="bg-yellow-100 text-yellow-800">{match}</mark>
        {after}
      </span>
    );
  };

  useEffect(() => {
    // If navigated here from an appointment, auto-open create modal with patient/appointment prefilled
    const params = new URLSearchParams(window.location.search);
    const patient_id = params.get('patient_id') || params.get('patientId');
    const appointment_id = params.get('appointment_id') || params.get('appointmentId');
    const from_appointment = params.get('from_appointment');
    const scroll_to = params.get('scroll_to');

    // If scroll_to provided, load and display that medical record detail
    if (scroll_to) {
      const recordId = Number(scroll_to);
      medicalRecordService.getMedicalRecordById(recordId)
        .then(res => {
          if (res && res.success && res.data) {
            setDetailRecord(res.data);
            setShowDetail(true);
          }
        })
        .catch(err => console.error('Error loading record detail:', err));
    }

    if (patient_id || appointment_id) {
      setForm(f => ({ ...f, patient_id: patient_id ? Number(patient_id) : undefined, appointment_id: appointment_id ? Number(appointment_id) : undefined, patientName: f.patientName || '' }));
      // If patient_id provided, load patient info for display in create modal
      if (patient_id) {
        const pid = Number(patient_id);
        patientService.getPatientById(pid)
          .then(res => {
            const payload = res?.data || res || {};
            const data = payload?.data || payload;
            setPatientInfo(data || null);
            // set form fields from patient info: id, name, and any available vitals as exam_results
            setForm(f => ({
              ...f,
              patient_id: pid,
              patientName: f.patientName || (data?.full_name || data?.user?.full_name || ''),
              exam_results: f.exam_results || (data?.latest_vitals || data?.last_vitals || null),
            }));
          })
          .catch(() => setPatientInfo(null));
      } else {
        setPatientInfo(null);
      }
      setShowModal(true);
      if (from_appointment) setRedirectAfterCreate(true);
    }

    fetchRecords();

    // cleanup debounce timer on unmount
    return () => {
      if (searchTimer.current) {
        clearTimeout(searchTimer.current);
        searchTimer.current = null;
      }
    };
  }, []);

  // If modal opens and we have a patient_id but no patientInfo, try to fetch it
  useEffect(() => {
    if (showModal && !patientInfo && form?.patient_id) {
      const pid = Number(form.patient_id);
      if (!pid) return;
      patientService.getPatientById(pid)
        .then(res => {
          const payload = res?.data || res || {};
          const data = payload?.data || payload;
          setPatientInfo(data || null);
        })
        .catch(() => setPatientInfo(null));
    }
  }, [showModal, form?.patient_id]);

  const fetchRecords = async (q = '') => {
    console.log('⟲ fetchRecords called with q=', q);
    setLoading(true);
    try {
      const res = await medicalRecordService.getMedicalRecords({ q });
      console.log('⟲ fetchRecords response:', res && res.success ? (res.data?.records?.length || 0) + ' records' : res);
      // res is expected to be { success: true, data: { records: [...], pagination: {...} } }
      if (res && res.success && res.data) {
        setRecords(res.data.records || []);
        setFilteredRecords(res.data.records || []);
      } else {
        setRecords([]);
        setFilteredRecords([]);
      }
    } catch (err) {
      console.error('Failed to load records', err);
      setRecords([]);
      setFilteredRecords([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    // immediate search when clicking the button
    if (searchTimer.current) {
      clearTimeout(searchTimer.current);
      searchTimer.current = null;
    }
    fetchRecords(query);
  };

  // live-search: called on each input change, debounced
  const handleQueryChange = (e) => {
    const q = e.target.value;
    console.log('↪ handleQueryChange value=', q);
    setQuery(q);
    // Immediate client-side filtering for snappy UX
    if (!q || !q.trim()) {
      setFilteredRecords(records);
    } else {
      const qLow = q.trim().toLowerCase();
      const local = (records || []).filter(r => {
        const name = (r.patient?.user?.full_name || r.patient?.full_name || r.patientName || '').toString().toLowerCase();
        const diagnosis = (r.diagnosis || '').toString().toLowerCase();
        const notes = (r.notes || '').toString().toLowerCase();
            const phone = (r.patient?.phone || r.patient?.user?.phone || '').toString().toLowerCase();
            // prefix match (startsWith) for quicker, predictable filtering
            return name.startsWith(qLow) || diagnosis.startsWith(qLow) || notes.startsWith(qLow) || phone.startsWith(qLow);
      });
      // if no prefix matches, fallback to contains to avoid showing empty result when user types a middle substring
      if ((!local || local.length === 0) && (records || []).length > 0) {
        const fallback = (records || []).filter(r => {
          const name = (r.patient?.user?.full_name || r.patient?.full_name || r.patientName || '').toString().toLowerCase();
          const diagnosis = (r.diagnosis || '').toString().toLowerCase();
          const notes = (r.notes || '').toString().toLowerCase();
          return name.includes(qLow) || diagnosis.includes(qLow) || notes.includes(qLow);
        });
        setFilteredRecords(fallback);
      } else {
        setFilteredRecords(local);
      }
    }

    if (searchTimer.current) clearTimeout(searchTimer.current);
    // debounce 350ms
    searchTimer.current = setTimeout(() => {
      fetchRecords(q);
      searchTimer.current = null;
    }, 350);
  };

  const openNew = () => {
    setEditing(null);
    setForm({ patientName: '', diagnosis: '', notes: '' });
    setPatientInfo(null);
    setShowModal(true);
  };

  const openEdit = (rec) => {
    setEditing(rec);
    setForm({ 
      patientName: rec.patient?.user?.full_name || rec.patientName || '', 
      diagnosis: rec.diagnosis, 
      notes: rec.notes,
      exam_results: rec.exam_results || null,
      lab_tests: rec.lab_tests || null,
    });
    // if record has patient_id, fetch patient info to show in form
    if (rec.patient_id) {
      patientService.getPatientById(rec.patient_id)
        .then(res => {
          const payload = res?.data || res || {};
          const data = payload?.data || payload;
          setPatientInfo(data || null);
          setForm(f => ({ 
            ...f, 
            patient_id: rec.patient_id, 
            patientName: f.patientName || (data?.full_name || data?.user?.full_name || ''),
          }));
        })
        .catch(() => setPatientInfo(null));
    }
    setShowModal(true);
  };

  const openDetail = async (rec) => {
    try {
      setLoading(true);
      const res = await medicalRecordService.getMedicalRecordById(rec.id);
      if (res && res.success && res.data) {
        // Primary record from API
        let record = res.data;
        setDetailRecord(record);
        setShowDetail(true);

        // If the record does not include prescriptions, try fallback queries
        try {
          const hasPres = record.prescriptions && Array.isArray(record.prescriptions) && record.prescriptions.length > 0;
          if (!hasPres) {
            // Try fetching prescriptions by medical_record_id
            const pRes = await prescriptionService.getPrescriptions({ medical_record_id: record.id });
            const pPayload = pRes?.data?.data || pRes?.data || pRes;
            let presList = [];
            if (Array.isArray(pPayload)) presList = pPayload;
            else if (Array.isArray(pPayload?.data)) presList = pPayload.data;
            else if (Array.isArray(pPayload?.prescriptions)) presList = pPayload.prescriptions;

            if (presList.length === 0 && record.patient_id) {
              // As a last resort, fetch recent prescriptions for the patient
              const pRes2 = await prescriptionService.getPrescriptions({ patient_id: record.patient_id, limit: 20 });
              const pPayload2 = pRes2?.data?.data || pRes2?.data || pRes2;
              if (Array.isArray(pPayload2)) presList = pPayload2;
              else if (Array.isArray(pPayload2?.data)) presList = pPayload2.data;
              else if (Array.isArray(pPayload2?.prescriptions)) presList = pPayload2.prescriptions;
            }

            if (presList && presList.length > 0) {
              record = { ...record, prescriptions: presList };
              setDetailRecord(record);
            }
          }
        } catch (err) {
          console.warn('Không thể tải đơn thuốc thay thế:', err);
        }
      } else {
        alert('Không thể tải chi tiết');
      }
    } catch (err) {
      console.error('Load detail failed', err);
      alert('Lỗi khi tải chi tiết');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Xóa hồ sơ này? Hành động không thể hoàn tác.')) return;
    try {
      const res = await medicalRecordService.deleteMedicalRecord(id);
      if (res && res.success) {
        setRecords(prev => prev.filter(r => r.id !== id));
      } else {
        throw new Error(res?.message || 'Delete failed');
      }
    } catch (err) {
      console.error('Delete failed', err);
      alert('Xóa thất bại: ' + (err?.response?.data?.message || err.message || 'Lỗi không xác định'));
    }
  };

  const handleSave = async (e) => {
    e.preventDefault();
    try {
      const payload = {
        appointment_id: form.appointment_id ? Number(form.appointment_id) : undefined,
        patient_id: form.patient_id ? Number(form.patient_id) : (patientInfo?.id ? Number(patientInfo.id) : undefined),
        diagnosis: form.diagnosis,
        notes: form.notes,
        attachments: form.attachments || null,
        exam_results: form.exam_results || null,
        lab_tests: form.lab_tests || null,
      };
      
      console.log('Gửi payload:', payload);
      
      let res;
      let savedRes;
      if (editing) {
        console.log('Cập nhật hồ sơ ID:', editing.id);
        savedRes = await medicalRecordService.updateMedicalRecord(editing.id, payload);
      } else {
        console.log('Tạo hồ sơ mới');
        savedRes = await medicalRecordService.createMedicalRecord(payload);
      }

      console.log('Response từ server:', savedRes);

      if (!savedRes || !savedRes.success) {
        console.error('Lỗi save:', savedRes);
        throw new Error(savedRes?.message || 'Save failed');
      }
      
      const saved = savedRes.data;
      console.log('Hồ sơ đã lưu:', saved);
      
      if (editing) {
        setRecords(prev => prev.map(r => (r.id === saved.id ? saved : r)));
      } else {
        setRecords(prev => [saved, ...prev]);
      }
      // Close the modal and refresh the list
      setShowModal(false);
      fetchRecords();
      alert('✅ Lưu hồ sơ thành công');
    } catch (err) {
      console.error('Lỗi chi tiết:', err);
      alert('Lưu thất bại: ' + (err?.response?.data?.message || err.message || 'Lỗi không xác định'));
    }
  };

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Hồ sơ bệnh án</h1>
          <p className="text-sm text-gray-500">Quản lý hồ sơ bệnh án: tìm kiếm, thêm, sửa, xóa</p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="Tìm theo tên bệnh nhân hoặc chẩn đoán"
              value={query}
              onChange={handleQueryChange}
              className="border rounded px-3 py-2 text-sm w-64"
            />
          </div>
          <button onClick={openNew} className="bg-blue-600 text-white px-4 py-2 rounded">Thêm hồ sơ</button>
        </div>
      </div>

      <div className="bg-white rounded shadow overflow-hidden">
        <table className="min-w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Bệnh nhân</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">SĐT</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Giới tính</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Ngày sinh</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Chẩn đoán</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Ghi chú</th>
              <th className="px-4 py-3 text-right text-sm font-medium text-gray-600">Hành động</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={4} className="p-6 text-center text-sm text-gray-500">Đang tải...</td></tr>
            ) : records.length === 0 ? (
              <tr><td colSpan={4} className="p-6 text-center text-sm text-gray-500">Không có hồ sơ</td></tr>
            ) : (
              // render filteredRecords when available
              (filteredRecords && filteredRecords.length > 0 ? filteredRecords : records).map(rec => {
                const nameVal = rec.patient?.user?.full_name || rec.patient?.full_name || rec.patientName || `Patient #${rec.patient_id || rec.id}`;
                const phoneVal = rec.patient?.phone || rec.patient?.user?.phone || '';
                const genderVal = rec.patient?.gender || '';
                const dobVal = rec.patient?.dob || rec.patient?.user?.dob || null;
                const diagVal = rec.diagnosis || '';
                const notesVal = rec.notes || '';
                return (
                 <tr key={rec.id} className="border-t hover:bg-gray-50">
                   <td className="px-4 py-3 text-sm">
                     {renderHighlighted(nameVal, query)}
                   </td>
                  <td className="px-4 py-3 text-sm">{renderHighlighted(phoneVal, query)}</td>
                  <td className="px-4 py-3 text-sm">{genderVal ? (genderVal === 'MALE' ? 'Nam' : genderVal === 'FEMALE' ? 'Nữ' : genderVal) : '—'}</td>
                  <td className="px-4 py-3 text-sm">{dobVal ? new Date(dobVal).toLocaleDateString('vi-VN') : '—'}</td>
                  <td className="px-4 py-3 text-sm">{renderHighlighted(diagVal, query)}</td>
                  <td className="px-4 py-3 text-sm truncate max-w-xs">{renderHighlighted(notesVal, query)}</td>
                  <td className="px-4 py-3 text-sm text-right">
                    <button onClick={() => openEdit(rec)} className="text-emerald-600 mr-3">Sửa</button>
                    <button onClick={() => openDetail(rec)} className="text-blue-600 mr-3">Xem</button>
                    {(user?.role?.name === 'ADMIN' || (user?.role?.name === 'DOCTOR' && (
                      // show delete when the record's doctor belongs to current user
                      rec.doctor?.user?.id === user?.id ||
                      ((user?.doctors?.[0]?.id || user?.doctor_id) === rec.doctor_id)
                    ))) && (
                      <button onClick={() => handleDelete(rec.id)} className="text-red-600">Xóa</button>
                    )}
                  </td>
                </tr>
              )
            })
            )}
          </tbody>
        </table>
      </div>

      {/* Modal */}
      {showModal &&
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-white w-full max-w-2xl rounded shadow-lg p-6">
            <h2 className="text-lg font-semibold mb-4">{editing ? 'Sửa hồ sơ' : 'Thêm hồ sơ mới'}</h2>
            <form onSubmit={handleSave} className="space-y-4">
              {/* If patientInfo exists (prefilled from appointment), show read-only patient card */}
              {patientInfo ? (
                <div className="p-3 bg-gray-50 rounded border">
                  <div className="text-sm text-gray-600">Bệnh nhân</div>
                  <div className="font-medium">{patientInfo.full_name || patientInfo.user?.full_name || form.patientName}</div>
                  <div className="text-xs text-gray-500">ID: {patientInfo.id}</div>
                  <div className="mt-2 text-sm text-gray-700">
                    <div><strong>Email:</strong> {patientInfo.user?.email || patientInfo.email || '—'}</div>
                    <div><strong>SĐT:</strong> {patientInfo.phone || patientInfo.user?.phone || '—'}</div>
                    <div>
                      <strong>Ngày sinh:</strong>{' '}
                      {patientInfo.dob || patientInfo.user?.dob ? new Date(patientInfo.dob || patientInfo.user?.dob).toLocaleDateString('vi-VN') : '—'}
                      {patientInfo.dob || patientInfo.user?.dob ? ` ( ${Math.floor((Date.now() - new Date(patientInfo.dob || patientInfo.user?.dob)) / 31557600000)} tuổi )` : ''}
                    </div>
                    <div><strong>Giới tính:</strong> {patientInfo.gender ? (patientInfo.gender === 'MALE' ? 'Nam' : patientInfo.gender === 'FEMALE' ? 'Nữ' : patientInfo.gender) : '—'}</div>
                    <div><strong>Địa chỉ:</strong> {(patientInfo.address || patientInfo.new_street || patientInfo.user?.address) || '—'}</div>
                    {patientInfo.identity_number && <div><strong>Số CMND/CCCD:</strong> {patientInfo.identity_number}</div>}
                  </div>
                </div>
              ) : (
                <div>
                  <label className="block text-sm text-gray-700">Tên bệnh nhân</label>
                  <input className="w-full border rounded px-3 py-2" value={form.patientName} onChange={e => setForm({...form, patientName: e.target.value})} />
                </div>
              )}
              <div>
                <label className="block text-sm text-gray-700">Chẩn đoán</label>
                <input className="w-full border rounded px-3 py-2" value={form.diagnosis} onChange={e => setForm({...form, diagnosis: e.target.value})} />
              </div>
              <div>
                <label className="block text-sm text-gray-700">Ghi chú</label>
                <textarea className="w-full border rounded px-3 py-2" rows={4} value={form.notes} onChange={e => setForm({...form, notes: e.target.value})} />
              </div>
              <div>
                <label className="block text-sm text-gray-700">Kết quả khám</label>
                <textarea 
                  className="w-full border rounded px-3 py-2" 
                  rows={3} 
                  placeholder="VD: Huyết áp: 120/80, Nhịp tim: 76 bpm, Nhiệt độ: 36.5°C"
                  value={form.exam_results || ''} 
                  onChange={e => setForm({...form, exam_results: e.target.value})} 
                />
              </div>
              <div>
                <label className="block text-sm text-gray-700">Xét nghiệm</label>
                <textarea 
                  className="w-full border rounded px-3 py-2" 
                  rows={3} 
                  placeholder="VD: Công thức máu, Sinh hóa, Nước tiểu, vv..."
                  value={form.lab_tests || ''} 
                  onChange={e => setForm({...form, lab_tests: e.target.value})} 
                />
              </div>

              <div className="flex justify-end gap-3">
                <button type="button" onClick={() => setShowModal(false)} className="px-4 py-2 border rounded">Hủy</button>
                <button type="submit" className="px-4 py-2 bg-emerald-600 text-white rounded">Lưu</button>
              </div>
            </form>
          </div>
        </div>
}

      {/* Detail Modal */}
      {showDetail && detailRecord &&
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-white w-full max-w-3xl rounded shadow-lg p-6 overflow-auto max-h-[80vh]">
              <div className="flex justify-between items-start">
              <div>
                <h2 className="text-xl font-semibold">Hồ sơ: {detailRecord.patient?.user?.full_name || detailRecord.patient?.full_name || detailRecord.patientName || `Patient #${detailRecord.patient_id}`}</h2>
                <p className="text-sm text-gray-500">Chẩn đoán: {detailRecord.diagnosis}</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    // navigate to prescription page with medical_record_id and patientId
                    const pid = detailRecord.patient_id || detailRecord.patient?.id || detailRecord.patient?.user?.id;
                    setShowDetail(false);
                    setDetailRecord(null);
                    navigate(`/doctor/prescription?medical_record_id=${detailRecord.id}${pid ? `&patientId=${pid}` : ''}`);
                  }}
                  className="px-3 py-1 bg-emerald-600 text-white rounded hover:bg-emerald-700"
                >
                  Kê thuốc
                </button>
                <button
                  onClick={async () => {
                    if (!detailRecord || !detailRecord.id) return;
                    if (!window.confirm('Gửi hồ sơ này cho bệnh nhân?')) return;
                    try {
                      setSending(true);
                      const res = await medicalRecordService.sendToPatient(detailRecord.id);
                      const payload = res?.data?.data || res?.data || res;
                      alert('✅ Đã gửi hồ sơ cho bệnh nhân');
                    } catch (err) {
                      console.error('Send to patient failed', err);
                      alert('Gửi thất bại: ' + (err?.response?.data?.message || err.message || 'Lỗi không xác định'));
                    } finally {
                      setSending(false);
                    }
                  }}
                  disabled={sending}
                  className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-60"
                >
                  {sending ? 'Đang gửi...' : 'Gửi cho bệnh nhân'}
                </button>
                <button onClick={() => { setShowDetail(false); setDetailRecord(null); }} className="px-3 py-1 border rounded hover:bg-gray-100">Đóng</button>
              </div>
            </div>

            <div className="mt-4 space-y-4">
              <div>
                <h3 className="font-medium">Ghi chú</h3>
                <div className="mt-1 text-sm text-gray-700 whitespace-pre-wrap">{detailRecord.notes}</div>
              </div>

              <div>
                <h3 className="font-medium">Kết quả khám</h3>
                <pre className="bg-gray-50 p-3 rounded text-sm text-gray-700 whitespace-pre-wrap">{detailRecord.exam_results || 'Chưa có'}</pre>
              </div>

              <div>
                <h3 className="font-medium">Đơn thuốc</h3>
                {detailRecord.prescriptions && Array.isArray(detailRecord.prescriptions) && detailRecord.prescriptions.length > 0 ? (
                  <div className="mt-2 space-y-2">
                    {detailRecord.prescriptions.map(p => {
                      // support items as either an array or a JSON-string (defensive)
                      let items = p.items;
                      try {
                        if (typeof items === 'string') items = JSON.parse(items);
                      } catch (e) {
                        items = [];
                      }

                      return (
                        <div key={p.id} className="border rounded p-3 bg-gray-50">
                          <div className="flex justify-between items-start mb-2">
                            <div>
                              <div className="text-sm font-medium">Đơn #{p.id}</div>
                              <div className="text-xs text-gray-500">
                                {p.created_at ? new Date(p.created_at).toLocaleDateString('vi-VN') : 'Chưa xác định'}
                              </div>
                            </div>
                            <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                              {p.status || 'DRAFT'}
                            </span>
                          </div>
                          {items && Array.isArray(items) && items.length > 0 ? (
                            <div className="mt-2 space-y-1">
                              {items.map((item, idx) => (
                                <div key={idx} className="text-sm text-gray-700 bg-white p-2 rounded">
                                  <div className="font-medium">{item.medicine_name || item.name || `Thuốc ${idx + 1}`}</div>
                                  <div className="text-xs text-gray-600 mt-1 space-y-0.5">
                                    {item.dosage && <div>Liều: {item.dosage}</div>}
                                    {item.quantity && <div>SL: {item.quantity}</div>}
                                    {item.instructions && <div>HD: {item.instructions}</div>}
                                    {item.unit_price && <div>Giá: {(item.unit_price || 0).toLocaleString('vi-VN')} đ</div>}
                                  </div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="text-xs text-gray-500 mt-1">Không có thông tin thuốc</div>
                          )}
                          {p.total_amount && (
                            <div className="text-sm font-medium text-gray-800 mt-2 pt-2 border-t">
                              Tổng: {p.total_amount.toLocaleString('vi-VN')} đ
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-sm text-gray-500 bg-gray-50 p-3 rounded mt-2">Không có đơn thuốc</div>
                )}
              </div>

              <div>
                <h3 className="font-medium">Xét nghiệm</h3>
                {detailRecord.lab_orders && detailRecord.lab_orders.length > 0 ? (
                  <div className="mt-2 space-y-2">
                    {detailRecord.lab_orders.map(l => (
                      <div key={l.id} className="border rounded p-3 bg-gray-50">
                        <div className="text-sm font-medium">Phiếu #{l.id} - {l.status}</div>
                        <div className="mt-2 text-sm text-gray-700 whitespace-pre-wrap">
                          {l.tests?.description || JSON.stringify(l.tests || {}, null, 2)}
                        </div>
                        {l.results && (
                          <div className="mt-2 text-sm text-gray-600">
                            <strong>Kết quả:</strong>
                            <pre className="bg-white p-2 rounded mt-1">{JSON.stringify(l.results, null, 2)}</pre>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-gray-500 bg-gray-50 p-3 rounded">Không có phiếu xét nghiệm liên quan</div>
                )}
              </div>
            </div>
          </div>
        </div>
      }
    </div>
  );
}
