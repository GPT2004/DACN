/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
import React, { useEffect, useState } from 'react';
import DoctorLayout from '../../components/doctor/DoctorLayout';
import { useAuth } from '../../context/AuthContext';
import { getDoctorAppointments, getTodayAppointments, getDoctorByUser } from '../../services/doctorService';
import { patientService } from '../../services/patientService';
import { prescriptionService } from '../../services/prescriptionService';
import { medicalRecordService } from '../../services/medicalRecordService';
import { useNavigate } from 'react-router-dom';
import PrescriptionForm from '../../components/doctor/PrescriptionForm';

export default function TodayAppointmentsPage() {
  const { user } = useAuth();
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedAppointment, setSelectedAppointment] = useState(null);
  const [patientDetails, setPatientDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [showPrescriptionModal, setShowPrescriptionModal] = useState(false);
  const [savingPrescription, setSavingPrescription] = useState(false);
  const [createdPrescription, setCreatedPrescription] = useState(null);
  const [showCreateRecordBeforePrescription, setShowCreateRecordBeforePrescription] = useState(false);
  const [quickRecordForm, setQuickRecordForm] = useState({ diagnosis: '', notes: '', exam_results: '', lab_tests: '' });
  const [creatingRecord, setCreatingRecord] = useState(false);
  const [currentMedicalRecordId, setCurrentMedicalRecordId] = useState(null);
  const [existingPrescriptions, setExistingPrescriptions] = useState([]);

  const doctorId = user?.doctors?.[0]?.id || user?.doctor_id || null;
  const navigate = useNavigate();

  // Robust formatter for TIME fields (handles HH:mm strings, Date objects, timestamps)
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

      // If no doctorId derived from user, try fetching doctor record by current user id
      if (!resolvedDoctorId && user?.id) {
        try {
          const r = await getDoctorByUser(user.id);
          // axios returns { data: { success, data, message } }
          const payload = r?.data || r || {};
          const doctor = payload?.data || payload;
          if (Array.isArray(doctor)) {
            resolvedDoctorId = doctor[0]?.id || null;
          } else {
            resolvedDoctorId = doctor?.id || null;
          }
        } catch (err) {
          console.warn('could not resolve doctor by user id', err);
        }
      }

      try {
        if (resolvedDoctorId) {
          // Get all upcoming appointments (not completed, cancelled, or no-show)
          const res = await getDoctorAppointments(resolvedDoctorId, { limit: 100 });
          const data = res?.data || res || {};
          const allAppointments = Array.isArray(data?.appointments) ? data.appointments : [];
          
          // Filter to only show CHECKED_IN and IN_PROGRESS appointments
          // (only appointments that have been checked in by reception)
          const upcomingAppointments = allAppointments.filter(apt => 
            ['CHECKED_IN', 'IN_PROGRESS'].includes(apt.status)
          );
          
          setAppointments(upcomingAppointments);
        } else {
          // Fallback to "me" endpoint if backend supports it
          const res = await getTodayAppointments({ limit: 100 });
          const data = res?.data || res || {};
          const allAppointments = data.appointments || data.appointments || data || [];
          const upcomingAppointments = Array.isArray(allAppointments) 
            ? allAppointments.filter(apt => ['CHECKED_IN', 'IN_PROGRESS'].includes(apt.status))
            : [];
          setAppointments(upcomingAppointments);
        }
      } catch (err) {
        console.error('load upcoming appointments', err);
      } finally {
        setLoading(false);
      }
    })();
  }, [doctorId]);

  // Fetch patient details when appointment is selected
  useEffect(() => {
    if (selectedAppointment?.patient_id) {
      setLoadingDetails(true);
      patientService.getPatientById(selectedAppointment.patient_id)
        .then(res => {
          const data = res?.data?.data || res?.data || res;
          setPatientDetails(data);
        })
        .catch(err => {
          console.error('Error loading patient details:', err);
          setPatientDetails(null);
        })
        .finally(() => setLoadingDetails(false));
    }
  }, [selectedAppointment]);

  if (loading) return <DoctorLayout><div className="text-center py-8">Đang tải...</div></DoctorLayout>;

  const handleSavePrescription = async (data) => {
    if (!selectedAppointment) return;
    try {
      setSavingPrescription(true);
      const patientId = selectedAppointment.patient_id || selectedAppointment.patient?.id || selectedAppointment.patient?.user?.id;
      const doctorId = selectedAppointment.doctor_id || selectedAppointment.doctor?.id;
      
      const items = data.medicines.map(m => ({
        medicine_id: m.medicine_id || undefined,
        medicine_name: m.name,
        quantity: Number(m.quantity) || 1,
        unit_price: m.unit_price || 0,
        instructions: m.instructions || '',
        dosage: m.dosage || ''
      }));

      const payload = {
        patient_id: Number(patientId),
        doctor_id: Number(doctorId),
        appointment_id: selectedAppointment.id,
        medical_record_id: currentMedicalRecordId ? Number(currentMedicalRecordId) : undefined,
        items,
      };

      const res = await prescriptionService.createPrescription(payload);
      const prescriptionData = res?.data?.data || res?.data || res;
      
      // Notify reception to create invoice
      if (prescriptionData?.id) {
        try {
          await prescriptionService.notifyReception(prescriptionData.id);
        } catch (err) {
          console.warn('Failed to notify reception:', err);
          // Don't fail the prescription creation if notification fails
        }
      }
      
      alert('✅ Lưu đơn thuốc thành công');
      
      // Show created prescription instead of closing
      setCreatedPrescription(prescriptionData);
      setShowPrescriptionModal(false);
    } catch (err) {
      alert('Lỗi lưu đơn thuốc: ' + (err?.response?.data?.message || err.message || ''));
    } finally {
      setSavingPrescription(false);
    }
  };

  const handleCreateRecordQuick = async () => {
    if (!selectedAppointment) return;
    try {
      setCreatingRecord(true);
      const patientId = selectedAppointment.patient_id || selectedAppointment.patient?.id || selectedAppointment.patient?.user?.id;
      const doctorId = selectedAppointment.doctor_id || selectedAppointment.doctor?.id;
      
      const res = await medicalRecordService.createMedicalRecord({
        patient_id: Number(patientId),
        doctor_id: Number(doctorId),
        appointment_id: selectedAppointment.id,
        diagnosis: quickRecordForm.diagnosis,
        notes: quickRecordForm.notes,
        exam_results: quickRecordForm.exam_results,
        lab_tests: quickRecordForm.lab_tests,
      });
      
      const newRecord = res?.data?.data || res?.data || res;
      if (newRecord?.id) {
        // Store the medical_record_id in state to use in prescription
        setCurrentMedicalRecordId(newRecord.id);
        setShowCreateRecordBeforePrescription(false);
        setQuickRecordForm({ diagnosis: '', notes: '', exam_results: '', lab_tests: '' });
        // Load existing prescriptions for this patient
        setExistingPrescriptions([]);
        setShowPrescriptionModal(true);
      }
    } catch (err) {
      alert('Lỗi tạo hồ sơ: ' + (err?.response?.data?.message || err.message || ''));
    } finally {
      setCreatingRecord(false);
    }
  };

  // Load existing prescriptions when prescription modal opens
  const handleOpenPrescriptionModal = async () => {
    setShowCreateRecordBeforePrescription(false);
    setQuickRecordForm({ diagnosis: '', notes: '', exam_results: '', lab_tests: '' });
    
    // Load all prescriptions of this patient (from all medical records)
    const patientId = selectedAppointment.patient_id || selectedAppointment.patient?.id || selectedAppointment.patient?.user?.id;
    if (patientId) {
      try {
        // Get all medical records for this patient to fetch their prescriptions
        const res = await medicalRecordService.getMedicalRecords({ 
          patientId: Number(patientId),
          limit: 100 
        });
        
        let allPrescriptions = [];
        if (res && res.data && res.data.records) {
          // Collect all prescriptions from all medical records
          res.data.records.forEach(record => {
            if (record.prescriptions && Array.isArray(record.prescriptions)) {
              allPrescriptions = allPrescriptions.concat(record.prescriptions);
            }
          });
        }
        
        setExistingPrescriptions(allPrescriptions || []);
      } catch (err) {
        console.error('Error loading prescriptions:', err);
        setExistingPrescriptions([]);
      }
    }
    
    setShowPrescriptionModal(true);
  };

  return (
    <DoctorLayout>
      <div>
        <h1 className="text-2xl font-bold mb-4">Lịch hẹn bệnh nhân</h1>
        {appointments.length === 0 ? (
          <div className="text-gray-500">Không có lịch hẹn chưa tới</div>
        ) : (
          <div className="space-y-3">
              {appointments.map(a => (
                  <div 
                    key={a.id} 
                    onClick={() => setSelectedAppointment(a)}
                    className="p-3 bg-white rounded shadow flex justify-between items-center cursor-pointer hover:bg-gray-50 transition"
                  >
                    <div>
                        <div className="font-medium">{a.patient?.full_name || a.patient?.user?.full_name || 'Bệnh nhân'}</div>
                      <div className="text-sm text-gray-500">
                        {a.appointment_date ? new Date(a.appointment_date).toLocaleDateString('vi-VN') + ' ' : ''}
                        {formatTimeHHmm(a.appointment_time || a.timeslot?.start_time)} — {a.status}
                      </div>
                      {/* Inline patient details always visible */}
                      <div className="text-sm text-gray-600 mt-2">
                        <div>ID: {a.patient_id || a.patient?.id || a.patient?.user?.id || '-'}</div>
                        <div>SĐT: {a.patient?.phone || a.patient?.user?.phone || '-'}</div>
                        { (a.patient?.dob || a.patient?.user?.dob) && (
                          <div>NS: {new Date(a.patient?.dob || a.patient?.user?.dob).toLocaleDateString('vi-VN')}</div>
                        )}
                        {a.reason && <div className="truncate">Lý do: {a.reason}</div>}
                      </div>
                    </div>
                    <div className="text-sm text-gray-500">Timeslot: {a.timeslot_id || '-'}</div>
                  </div>
                ))}
          </div>
        )}
      </div>

      {/* Modal Chi Tiết Lịch Hẹn */}
      {selectedAppointment && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-lg max-w-md w-full p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Chi tiết lịch hẹn</h2>
              <button
                onClick={() => setSelectedAppointment(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>

            {loadingDetails ? (
              <div className="text-center py-8">Đang tải thông tin...</div>
            ) : (
              <div className="space-y-4">
                {/* Thông tin bệnh nhân */}
                <div className="border-b pb-4">
                  <h3 className="font-semibold text-gray-700 mb-2">Thông tin bệnh nhân</h3>
                  <div className="space-y-2 text-sm">
                      <div>
                        <span className="text-gray-600">Tên:</span>
                        <span className="font-medium ml-2">{patientDetails?.full_name || selectedAppointment.patient?.full_name || selectedAppointment.patient?.user?.full_name || '-'}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Mã bệnh nhân:</span>
                        <span className="font-medium ml-2">{patientDetails?.id || selectedAppointment.patient_id || selectedAppointment.patient?.id || '-'}</span>
                      </div>
                    {patientDetails?.user?.dob && (
                      <div>
                        <span className="text-gray-600">Năm sinh:</span>
                        <span className="font-medium ml-2">
                            {new Date(patientDetails.dob || patientDetails.user?.dob).getFullYear()}
                        </span>
                      </div>
                    )}
                    {patientDetails?.gender && (
                      <div>
                        <span className="text-gray-600">Giới tính:</span>
                        <span className="font-medium ml-2">
                          {patientDetails.gender === 'MALE' ? 'Nam' : patientDetails.gender === 'FEMALE' ? 'Nữ' : patientDetails.gender}
                        </span>
                      </div>
                    )}
                      {(patientDetails?.phone || selectedAppointment.patient?.phone || selectedAppointment.patient?.user?.phone) && (
                        <div>
                          <span className="text-gray-600">Điện thoại:</span>
                          <span className="font-medium ml-2">{patientDetails?.phone || selectedAppointment.patient?.phone || selectedAppointment.patient?.user?.phone}</span>
                        </div>
                      )}
                  </div>
                </div>

                {/* Thông tin lịch khám */}
                <div className="border-b pb-4">
                  <h3 className="font-semibold text-gray-700 mb-2">Thông tin lịch khám</h3>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-gray-600">Ngày:</span>
                      <span className="font-medium ml-2">
                        {selectedAppointment.appointment_date 
                          ? new Date(selectedAppointment.appointment_date).toLocaleDateString('vi-VN')
                          : '-'
                        }
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Giờ:</span>
                      <span className="font-medium ml-2">
                        {formatTimeHHmm(selectedAppointment.appointment_time || selectedAppointment.timeslot?.start_time)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Trạng thái:</span>
                      <span className="font-medium ml-2 px-2 py-1 bg-blue-100 text-blue-800 rounded">
                        {selectedAppointment.status}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Lý do khám */}
                {selectedAppointment.reason && (
                  <div>
                    <h3 className="font-semibold text-gray-700 mb-2">Lý do khám</h3>
                    <p className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                      {selectedAppointment.reason}
                    </p>
                  </div>
                )}
              </div>
            )}

            <div className="mt-6 flex gap-2">
              <button
                onClick={() => {
                  const q = new URLSearchParams();
                  const patientId = selectedAppointment.patient_id || selectedAppointment.patient?.id || selectedAppointment.patient?.user?.id;
                  if (patientId) q.set('patient_id', String(patientId));
                  q.set('appointment_id', String(selectedAppointment.id));
                  q.set('from_appointment', '1');
                  setSelectedAppointment(null);
                  navigate(`/doctor/medical-records?${q.toString()}`);
                }}
                className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 transition"
              >
                Thêm hồ sơ
              </button>
              <button
                onClick={handleOpenPrescriptionModal}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
              >
                Kê thuốc
              </button>
              <button
                onClick={() => setSelectedAppointment(null)}
                className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400 transition"
              >
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Quick Create Medical Record Modal - Before Prescription */}
      {showCreateRecordBeforePrescription && selectedAppointment && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50 overflow-auto">
          <div className="bg-white rounded-lg shadow-lg max-w-2xl w-full my-8">
            <div className="flex justify-between items-center p-6 border-b">
              <h2 className="text-xl font-bold">Tạo hồ sơ bệnh án để kê thuốc</h2>
              <button
                onClick={() => setShowCreateRecordBeforePrescription(false)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            <form onSubmit={(e) => { e.preventDefault(); handleCreateRecordQuick(); }} className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Chẩn đoán</label>
                <input 
                  className="w-full border rounded px-3 py-2" 
                  placeholder="VD: Cảm cúm"
                  value={quickRecordForm.diagnosis}
                  onChange={e => setQuickRecordForm({...quickRecordForm, diagnosis: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Ghi chú</label>
                <textarea 
                  className="w-full border rounded px-3 py-2" 
                  rows={2}
                  placeholder="Ghi chú bổ sung..."
                  value={quickRecordForm.notes}
                  onChange={e => setQuickRecordForm({...quickRecordForm, notes: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Kết quả khám</label>
                <textarea 
                  className="w-full border rounded px-3 py-2" 
                  rows={2}
                  placeholder="VD: Huyết áp 120/80, Nhiệt độ 36.5°C"
                  value={quickRecordForm.exam_results}
                  onChange={e => setQuickRecordForm({...quickRecordForm, exam_results: e.target.value})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Xét nghiệm</label>
                <textarea 
                  className="w-full border rounded px-3 py-2" 
                  rows={2}
                  placeholder="VD: Công thức máu, Sinh hóa..."
                  value={quickRecordForm.lab_tests}
                  onChange={e => setQuickRecordForm({...quickRecordForm, lab_tests: e.target.value})}
                />
              </div>

              <div className="flex gap-2 justify-end pt-4 border-t">
                <button
                  type="button"
                  onClick={() => setShowCreateRecordBeforePrescription(false)}
                  className="px-4 py-2 border rounded hover:bg-gray-100"
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  disabled={creatingRecord}
                  className="px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50"
                >
                  {creatingRecord ? 'Đang tạo...' : 'Tạo hồ sơ & Tiếp tục kê thuốc'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Prescription Modal */}
      {showPrescriptionModal && selectedAppointment && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50 overflow-auto">
          <div className="bg-white rounded-lg shadow-lg max-w-2xl w-full my-8">
            <div className="flex justify-between items-center p-6 border-b">
              <h2 className="text-xl font-bold">Kê đơn thuốc</h2>
              <button
                onClick={() => {
                  setShowPrescriptionModal(false);
                  setCurrentMedicalRecordId(null);
                }}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            <div className="p-6 max-h-[70vh] overflow-auto">
              <PrescriptionForm 
                patient={{ name: `ID: ${selectedAppointment.patient_id || selectedAppointment.patient?.id || selectedAppointment.patient?.user?.id} - ${selectedAppointment.patient?.full_name || selectedAppointment.patient?.user?.full_name || 'Bệnh nhân'}` }}
                onSave={handleSavePrescription}
                onCancel={() => {
                  setShowPrescriptionModal(false);
                  setCurrentMedicalRecordId(null);
                  setExistingPrescriptions([]);
                }}
                initialPrescriptions={existingPrescriptions}
              />
            </div>
          </div>
        </div>
      )}

      {/* Prescription Result Modal - Show after creation */}
      {createdPrescription && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50 overflow-auto">
          <div className="bg-white rounded-lg shadow-lg max-w-2xl w-full my-8">
            <div className="flex justify-between items-center p-6 border-b">
              <h2 className="text-xl font-bold">✅ Đơn thuốc đã lưu</h2>
              <button
                onClick={() => {
                  setCreatedPrescription(null);
                  setSelectedAppointment(null);
                  setAppointments(prev => prev.filter(a => a.id !== selectedAppointment.id));
                  navigate('/doctor/dashboard');
                }}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            <div className="p-6 max-h-[70vh] overflow-auto">
              <div className="space-y-6">
                {/* Thông tin cơ bản */}
                <div className="border-b pb-4">
                  <h3 className="font-semibold text-gray-700 mb-3">Thông tin đơn thuốc</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Mã đơn:</span>
                      <span className="font-medium ml-2">#{createdPrescription?.id}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Trạng thái:</span>
                      <span className="font-medium ml-2 px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                        {createdPrescription?.status || 'ACTIVE'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Bệnh nhân:</span>
                      <span className="font-medium ml-2">ID {createdPrescription?.patient_id}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Ngày tạo:</span>
                      <span className="font-medium ml-2">
                        {createdPrescription?.created_at 
                          ? new Date(createdPrescription.created_at).toLocaleDateString('vi-VN')
                          : 'Hôm nay'
                        }
                      </span>
                    </div>
                  </div>
                </div>

                {/* Danh sách thuốc */}
                <div className="border-b pb-4">
                  <h3 className="font-semibold text-gray-700 mb-3">Các loại thuốc</h3>
                  <div className="space-y-2">
                    {createdPrescription?.items && Array.isArray(createdPrescription.items) ? (
                      createdPrescription.items.map((item, idx) => (
                        <div key={idx} className="bg-gray-50 p-3 rounded border-l-2 border-blue-500">
                          <div className="font-medium text-gray-800">{item.medicine_name || item.name || `Thuốc ${idx + 1}`}</div>
                          <div className="text-sm text-gray-600 mt-1 space-y-1">
                            {item.dosage && <div>Liều lượng: <span className="font-medium">{item.dosage}</span></div>}
                            {item.quantity && <div>Số lượng: <span className="font-medium">{item.quantity}</span></div>}
                            {item.instructions && <div>Hướng dẫn: <span className="font-medium">{item.instructions}</span></div>}
                            {item.unit_price && <div>Giá: <span className="font-medium">{(item.unit_price || 0).toLocaleString('vi-VN')} đ</span></div>}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-gray-500 text-sm">Không có thông tin thuốc</div>
                    )}
                  </div>
                </div>

                {/* Ghi chú */}
                {createdPrescription?.notes && (
                  <div className="border-b pb-4">
                    <h3 className="font-semibold text-gray-700 mb-2">Ghi chú</h3>
                    <p className="text-sm text-gray-600 bg-yellow-50 p-3 rounded">
                      {createdPrescription.notes}
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="p-6 border-t flex gap-2 justify-end">
              <button
                onClick={() => {
                  // Print or download logic here
                  window.print && window.print();
                }}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition"
              >
                In đơn
              </button>
              <button
                onClick={() => {
                  // Navigate to medical records page to view full detail with prescription
                  if (currentMedicalRecordId) {
                    setCreatedPrescription(null);
                    setSelectedAppointment(null);
                    setAppointments(prev => prev.filter(a => a.id !== selectedAppointment.id));
                    setCurrentMedicalRecordId(null);
                    navigate(`/doctor/medical-records?scroll_to=${currentMedicalRecordId}`);
                  }
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
              >
                Xem chi tiết hồ sơ
              </button>
              <button
                onClick={() => {
                  setCreatedPrescription(null);
                  setSelectedAppointment(null);
                  setAppointments(prev => prev.filter(a => a.id !== selectedAppointment.id));
                  setCurrentMedicalRecordId(null);
                  navigate('/doctor/dashboard');
                }}
                className="px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 transition"
              >
                Xong, về Dashboard
              </button>
            </div>
          </div>
        </div>
      )}
    </DoctorLayout>
  );
}
  // file ends here
