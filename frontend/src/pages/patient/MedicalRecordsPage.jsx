import React, { useState, useEffect, useRef } from 'react';
import PublicHeader from '../../components/common/PublicHeader';
import PatientPageHeader from '../../components/patient/PatientPageHeader';
import Footer from '../../components/patient/clinic/Footer';
import { FileText, Download, Eye } from 'lucide-react';
import { patientService } from '../../services/patientService';
import { useAuth } from '../../context/AuthContext';
import { medicalRecordService } from '../../services/medicalRecordService';
import { useNavigate } from 'react-router-dom';

export default function MedicalRecordsPage() {
  const [records, setRecords] = useState([]);
  const [filteredRecords, setFilteredRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  const searchTimer = useRef(null);
  const { user } = useAuth();
  const navigate = useNavigate();
  const [showDetail, setShowDetail] = useState(false);
  const [detailRecord, setDetailRecord] = useState(null);

  useEffect(() => {
    let mounted = true;

    const loadRecords = async () => {
      try {
        setLoading(true);
        setError(null);

        // First get the patient profile for current user
        const profileRes = await patientService.getMyProfile();
        const patient = profileRes?.data?.data || profileRes?.data || null;
        const patientId = patient?.id;

        if (!patientId) {
          // If no patient profile, nothing to load
          if (mounted) {
            setRecords([]);
            setFilteredRecords([]);
            setLoading(false);
          }
          return;
        }

        const res = await patientService.getPatientMedicalRecords(patientId, { page: 1, limit: 200 });
        const data = res?.data?.data || res?.data || null;
        if (mounted) {
          const recs = Array.isArray(data?.records) ? data.records : [];
          setRecords(recs);
          setFilteredRecords(recs);
        }
      } catch (err) {
        console.error('Failed to load medical records:', err);
        if (mounted) setError(err?.response?.data?.message || err.message || 'Lỗi khi tải hồ sơ');
      } finally {
        if (mounted) setLoading(false);
      }
    };

    loadRecords();

    return () => { mounted = false; };
  }, [user]);

  // Highlight helper (same as doctor page)
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

  // Client-side live filtering (prefix-first, contains fallback)
  const handleQueryChange = (e) => {
    const q = e.target.value;
    setQuery(q);
    if (!q || !q.trim()) {
      setFilteredRecords(records);
    } else {
      const qLow = q.trim().toLowerCase();
      const local = (records || []).filter(r => {
        const doctorName = (r.doctor?.user?.full_name || r.doctor?.full_name || r.doctor || '').toString().toLowerCase();
        const diagnosis = (r.diagnosis || '').toString().toLowerCase();
        const notes = (r.notes || '').toString().toLowerCase();
        return doctorName.startsWith(qLow) || diagnosis.startsWith(qLow) || notes.startsWith(qLow);
      });
      if ((!local || local.length === 0) && (records || []).length > 0) {
        const fallback = (records || []).filter(r => {
          const doctorName = (r.doctor?.user?.full_name || r.doctor?.full_name || r.doctor || '').toString().toLowerCase();
          const diagnosis = (r.diagnosis || '').toString().toLowerCase();
          const notes = (r.notes || '').toString().toLowerCase();
          return doctorName.includes(qLow) || diagnosis.includes(qLow) || notes.includes(qLow);
        });
        setFilteredRecords(fallback);
      } else {
        setFilteredRecords(local);
      }
    }

    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => {
      // for patient page we keep search client-side; could call server if needed
      searchTimer.current = null;
    }, 350);
  };

  const openDetail = async (rec) => {
    try {
      setLoading(true);
      const res = await medicalRecordService.getMedicalRecordById(rec.id);
      const detail = res?.data?.data || res?.data || res;
      if (detail && detail.id) {
        // preserve shared metadata from list result so modal can show it
        if (rec.shared) {
          detail.shared = true;
          detail.shared_at = rec.shared_at || null;
          detail.shared_by_user_id = rec.shared_by_user_id || null;
        }
        setDetailRecord(detail);
        setShowDetail(true);
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

  return (
    <div className="flex flex-col min-h-screen">
      <PublicHeader />
      <PatientPageHeader title="📋 Hồ sơ bệnh án" description="Lịch sử khám và các hồ sơ bệnh án của bạn" />
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">

        {loading ? (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4 animate-pulse" />
            <h3 className="text-xl font-semibold text-gray-700 mb-2">Đang tải hồ sơ...</h3>
            <p className="text-gray-500">Vui lòng chờ trong giây lát.</p>
          </div>
        ) : error ? (
          <div className="bg-white rounded-lg shadow-md p-6 text-center">
            <p className="text-red-600 font-semibold">{error}</p>
          </div>
        ) : records.length === 0 ? (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-700 mb-2">Không có hồ sơ nào</h3>
            <p className="text-gray-500">Bạn chưa có hồ sơ bệnh án nào.</p>
          </div>
        ) : (
          <div>
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-2xl font-bold">Hồ sơ bệnh án</h1>
                <p className="text-sm text-gray-500">Lịch sử khám của bạn</p>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    placeholder="Tìm theo tên bác sĩ hoặc chẩn đoán"
                    value={query}
                    onChange={handleQueryChange}
                    className="border rounded px-3 py-2 text-sm w-64"
                  />
                </div>
              </div>
            </div>

            <div className="bg-white rounded shadow overflow-hidden">
              <table className="min-w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Bác sĩ</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Ngày khám</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Chẩn đoán</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Hồ sơ dùng</th>
                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">Ghi chú</th>
                    <th className="px-4 py-3 text-right text-sm font-medium text-gray-600">Hành động</th>
                  </tr>
                </thead>
                <tbody>
                  {loading ? (
                    <tr><td colSpan={5} className="p-6 text-center text-sm text-gray-500">Đang tải...</td></tr>
                  ) : (filteredRecords && filteredRecords.length > 0 ? filteredRecords : records).map(rec => {
                    const doctorName = rec.doctor?.user?.full_name || rec.doctor?.full_name || rec.doctor || '-';
                    const dateVal = rec.appointment?.appointment_date || rec.created_at || rec.date || null;
                    const diagVal = rec.diagnosis || '';
                    const notesVal = rec.notes || '';
                    const profileName = rec.patient?.user?.full_name || rec.patient?.full_name || '';
                    const profilePhone = rec.patient?.user?.phone || rec.patient?.phone || '';
                    return (
                      <tr key={rec.id} className="border-t hover:bg-gray-50">
                        <td className="px-4 py-3 text-sm">
                          <div className="flex items-center gap-2">
                            <div>{renderHighlighted(doctorName, query)}</div>
                            {rec.shared && (
                              <span className="text-xs px-2 py-0.5 bg-green-100 text-green-800 rounded">Được chia sẻ</span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm">{dateVal ? new Date(dateVal).toLocaleDateString('vi-VN') : '—'}</td>
                        <td className="px-4 py-3 text-sm">{renderHighlighted(diagVal, query)}</td>
                        <td className="px-4 py-3 text-sm">
                          {profileName || '—'}
                          {profilePhone ? <div className="text-xs text-gray-500">{profilePhone}</div> : null}
                        </td>
                        <td className="px-4 py-3 text-sm truncate max-w-xs">{renderHighlighted(notesVal, query)}</td>
                        <td className="px-4 py-3 text-sm text-right">
                          <button onClick={() => openDetail(rec)} className="text-blue-600 mr-3">Xem</button>
                          <button className="text-gray-600">Tải</button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}  
      </main>
        {/* Detail Modal */}
        {showDetail && detailRecord && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
            <div className="bg-white w-full max-w-3xl rounded shadow-lg p-6 overflow-auto max-h-[80vh]">
              <div className="flex justify-between items-start">
                <div>
                  <div className="flex items-center gap-3">
                    <div>
                      <h2 className="text-xl font-semibold">Hồ sơ: {detailRecord.patient?.user?.full_name || detailRecord.patient?.full_name || detailRecord.patientName || `Patient #${detailRecord.patient_id}`}</h2>
                      <p className="text-sm text-gray-500">Chẩn đoán: {detailRecord.diagnosis}</p>
                    </div>
                    {detailRecord.shared && (
                      <div className="text-sm">
                        <span className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded">Được chia sẻ</span>
                        {detailRecord.shared_at && (
                          <div className="text-xs text-gray-500 mt-1">{new Date(detailRecord.shared_at).toLocaleString('vi-VN')}</div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
                <div>
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
                        let items = p.items;
                        try { if (typeof items === 'string') items = JSON.parse(items); } catch (e) { items = []; }
                        return (
                          <div key={p.id} className="border rounded p-3 bg-gray-50">
                            <div className="flex justify-between items-start mb-2">
                              <div>
                                <div className="text-sm font-medium">Đơn #{p.id}</div>
                                <div className="text-xs text-gray-500">{p.created_at ? new Date(p.created_at).toLocaleDateString('vi-VN') : 'Chưa xác định'}</div>
                              </div>
                              <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">{p.status || 'DRAFT'}</span>
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
                              <div className="text-sm font-medium text-gray-800 mt-2 pt-2 border-t">Tổng: {p.total_amount.toLocaleString('vi-VN')} đ</div>
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
                          <div className="mt-2 text-sm text-gray-700 whitespace-pre-wrap">{l.tests?.description || JSON.stringify(l.tests || {}, null, 2)}</div>
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
        )}

        <Footer />
    </div>
  );
}
