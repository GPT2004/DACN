/* eslint-disable no-console */
/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-unused-vars */
// frontend/src/components/patient/BookAppointmentForm.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Calendar, Clock, User, FileText, Search } from 'lucide-react';
import Input from '../common/Input';
import Button from '../common/Button';
import Alert from '../common/Alert';
import { timeslotService } from '../../services/scheduleService';

export default function BookAppointmentForm({ 
  onSubmit, 
  loading = false,
  doctors = [],
  onLoadDoctors,
  patients = [],
  onLoadPatients,
  initialValues = null,
  allowNewPatient = false
}) {
  const [formData, setFormData] = useState({
    doctor_id: '',
    patient_id: '',
    appointment_date: '',
    appointment_time: '',
    timeslot_id: '',
    reason: '',
    notes: '',
  });
  const [errors, setErrors] = useState({});
  const [newPatientMode, setNewPatientMode] = useState(false);
  const [creatingPatient, setCreatingPatient] = useState(false);
  const [showRegisterRedirect, setShowRegisterRedirect] = useState(false);
  const navigate = useNavigate();
  const [newPatientData, setNewPatientData] = useState({ full_name: '', phone: '', email: '', dob: '' });
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [availableSlots, setAvailableSlots] = useState([]);
  const [specialtyFilter, setSpecialtyFilter] = useState('');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    if (onLoadDoctors) {
      onLoadDoctors();
    }
    if (onLoadPatients) {
      onLoadPatients();
    }
  }, [onLoadDoctors, onLoadPatients]);

  // Apply initial values when provided (e.g., prefill from ScheduleModal)
  useEffect(() => {
    if (initialValues) {
      setFormData(prev => ({ ...prev, ...initialValues }));
      if (initialValues.doctor_id && !selectedDoctor) {
        const dr = doctors.find(d => d.id === initialValues.doctor_id);
        if (dr) setSelectedDoctor(dr);
      }
    }
  }, [initialValues]);

  // Generate time slots
  const generateTimeSlots = () => {
    const slots = [];
    const startHour = 7;
    const endHour = 20;
    
    for (let hour = startHour; hour < endHour; hour++) {
      for (let minute = 0; minute < 60; minute += 20) {
        const timeStr = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
        slots.push(timeStr);
      }
    }
    
    return slots;
  };

  useEffect(() => {
    setAvailableSlots([]);
  }, []);

  // Load timeslots for selected doctor + date
  useEffect(() => {
    const loadSlots = async () => {
      const doctorId = formData.doctor_id;
      const date = formData.appointment_date;
      if (!doctorId || !date) {
        setAvailableSlots([]);
        return;
      }

      // clear previously selected timeslot when doctor/date changes
      setFormData(prev => ({ ...prev, timeslot_id: '', appointment_time: '' }));

      try {
        const res = await timeslotService.getAvailableTimeslots(doctorId, date);
        const list = res?.data?.data || res?.data || [];
        // Normalize to array of { id, label, start_time }
        const mapped = Array.isArray(list) ? list.map(ts => {
          // try multiple shapes
          const start = ts.start_time || ts.time || ts.start || (ts.start_time_str) || '';
          const end = ts.end_time || ts.end || ts.end_time_str || '';
          let label = start;
          if (start && end) label = `${start} - ${end}`;
          return { id: ts.id || ts.timeslot_id || ts.timeslotId, label, raw: ts };
        }) : [];
        setAvailableSlots(mapped);
      } catch (err) {
        console.error('Failed to load timeslots:', err?.response || err);
        setAvailableSlots([]);
      }
    };

    loadSlots();
  }, [formData.doctor_id, formData.appointment_date]);

  const filteredDoctors = doctors.filter(doctor => {
    const matchesSearch = doctor.user?.full_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         doctor.specialty?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSpecialty = !specialtyFilter || doctor.specialty === specialtyFilter;
    return matchesSearch && matchesSpecialty;
  });

  const specialties = [...new Set(doctors.map(d => d.specialty).filter(Boolean))];
  const filteredPatients = patients.filter(p => p.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) || p.user?.email?.toLowerCase().includes(searchTerm.toLowerCase()));

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const handleDoctorSelect = (doctor) => {
    setSelectedDoctor(doctor);
    handleChange('doctor_id', doctor.id);
  };

  const handlePatientSelect = (patient) => {
    handleChange('patient_id', patient.id);
  };

  const handleNewPatientChange = (field, value) => {
    setNewPatientData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) setErrors(prev => ({ ...prev, [field]: '' }));
  };

  const validateForm = () => {
    const newErrors = {};

    // If receptionist is creating a new patient inline, skip patient_id validation here
    if (!(newPatientMode && allowNewPatient) && !formData.patient_id) {
      newErrors.patient_id = 'Vui lòng chọn bệnh nhân';
    }
    if (!formData.doctor_id) {
      newErrors.doctor_id = 'Vui lòng chọn bác sĩ';
    }
    if (!formData.appointment_date) {
      newErrors.appointment_date = 'Vui lòng chọn ngày khám';
    }
      // Require a selected timeslot from schedule. If no timeslots are available for the chosen date, show an error.
      if (!formData.timeslot_id) {
        if (availableSlots.length === 0) {
          newErrors.appointment_time = 'Bác sĩ không có lịch làm việc ngày đã chọn';
        } else {
          newErrors.appointment_time = 'Vui lòng chọn khung giờ từ lịch làm việc của bác sĩ';
        }
      }
    if (!formData.reason || formData.reason.trim().length < 10) {
      newErrors.reason = 'Vui lòng nhập lý do khám (tối thiểu 10 ký tự)';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    // If receptionist chose to create a new patient, create patient first
    try {
      let payload = { ...formData };

      if (newPatientMode && allowNewPatient) {
        // basic validation for new patient
        if (!newPatientData.full_name || !newPatientData.phone) {
          setErrors(prev => ({ ...prev, full_name: newPatientData.full_name ? '' : 'Họ tên bắt buộc', phone: newPatientData.phone ? '' : 'Số điện thoại bắt buộc' }));
          return;
        }

        setCreatingPatient(true);
        // prevent double-submit while creating
        setErrors(prev => ({ ...prev, form: '' }));
        const { patientService } = await import('../../services/patientService');
        // generate a random password since backend requires it
        const password = `p@tient${Math.floor(Math.random() * 90000) + 10000}`;
        const patientPayload = {
          full_name: newPatientData.full_name,
          phone: newPatientData.phone,
          email: newPatientData.email || `patient+${Date.now()}@example.test`,
          dob: newPatientData.dob || undefined,
          password,
        };

        const res = await patientService.createPatient(patientPayload);
        const created = res?.data?.data || res?.data || null;
        setCreatingPatient(false);
        if (!created || !created.id) {
          throw new Error('Không thể tạo bệnh nhân');
        }

        // set created patient id into payload
        payload.patient_id = created.id;
      }

      // Ensure timeslot_id is numeric if present (backend expects number)
      if (payload.timeslot_id) payload.timeslot_id = Number(payload.timeslot_id);

      if (onSubmit) {
        await onSubmit(payload);
      }
    } catch (err) {
      setCreatingPatient(false);
      console.error('Error creating new patient or submitting appointment:', err?.response || err);
      // Handle 403 specifically (insufficient permissions)
      const status = err?.response?.status;
      if (status === 403) {
        setErrors(prev => ({ ...prev, form: 'Không đủ quyền để tạo bệnh nhân. Vui lòng tạo bệnh nhân trước hoặc liên hệ Admin.' }));
        // Suggest opening the registration page for receptionist
        setShowRegisterRedirect(true);
        return;
      }

      setErrors(prev => ({ ...prev, form: err?.response?.data?.message || err.message || 'Lỗi khi tạo bệnh nhân/đặt lịch' }));
    }
  };

  const minDate = (() => {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  })();
  
  const maxDate = (() => {
    const future = new Date();
    future.setDate(future.getDate() + 30);
    const year = future.getFullYear();
    const month = String(future.getMonth() + 1).padStart(2, '0');
    const day = String(future.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  })();

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Step 0: Select Patient (Receptionist) */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <User className="w-5 h-5 text-blue-600" />
          Bước 0: Chọn Bệnh nhân
        </h3>

        <div className="mb-3">
          <input
            type="text"
            placeholder="Tìm bệnh nhân theo tên hoặc email..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        {/* Toggle between existing patient and new patient */}
        {allowNewPatient && (
          <div className="flex items-center gap-3 mb-3">
            <button type="button" onClick={() => setNewPatientMode(false)} className={`px-3 py-1 rounded ${!newPatientMode ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}>Chọn bệnh nhân có sẵn</button>
            <button type="button" onClick={() => setNewPatientMode(true)} className={`px-3 py-1 rounded ${newPatientMode ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}>Thêm bệnh nhân mới</button>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-56 overflow-y-auto mb-2">
          {!newPatientMode && filteredPatients.map(p => (
            <div key={p.id} onClick={() => handlePatientSelect(p)} className={`p-3 border rounded-lg cursor-pointer ${formData.patient_id === p.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-300'}`}>
              <div className="font-medium text-gray-900">{p.user?.full_name}</div>
              <div className="text-xs text-gray-500">{p.user?.email || ''}</div>
            </div>
          ))}

          {/* New patient form inside the selector area when in newPatientMode */}
          {newPatientMode && (
            <div className="p-3 border rounded-lg bg-gray-50">
              <div className="space-y-3">
                <Input label="Họ và tên" value={newPatientData.full_name} onChange={(e) => handleNewPatientChange('full_name', e.target.value)} />
                <Input label="Số điện thoại" value={newPatientData.phone} onChange={(e) => handleNewPatientChange('phone', e.target.value)} />
                <Input label="Email (tùy chọn)" value={newPatientData.email} onChange={(e) => handleNewPatientChange('email', e.target.value)} />
                <Input label="Ngày sinh" type="date" value={newPatientData.dob} onChange={(e) => handleNewPatientChange('dob', e.target.value)} />
                {errors.full_name && <p className="text-red-500 text-sm">{errors.full_name}</p>}
                {errors.phone && <p className="text-red-500 text-sm">{errors.phone}</p>}
              </div>
            </div>
          )}
        </div>
        {errors.patient_id && <p className="text-red-500 text-sm mt-2">{errors.patient_id}</p>}
        {errors.form && <p className="text-red-500 text-sm mt-2">{errors.form}</p>}
      </div>

      {/* Step 1: Select Doctor */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <User className="w-5 h-5 text-blue-600" />
          Bước 1: Chọn Bác sĩ
        </h3>

        {/* Search and Filter */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Tìm kiếm bác sĩ, chuyên khoa..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <select
            value={specialtyFilter}
            onChange={(e) => setSpecialtyFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Tất cả chuyên khoa</option>
            {specialties.map(specialty => (
              <option key={specialty} value={specialty}>{specialty}</option>
            ))}
          </select>
        </div>

        {/* Doctor List */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-96 overflow-y-auto">
          {filteredDoctors.map(doctor => (
            <div
              key={doctor.id}
              onClick={() => handleDoctorSelect(doctor)}
              className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                selectedDoctor?.id === doctor.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300'
              }`}
            >
              <div className="flex items-start gap-3">
                <img
                  src={doctor.user?.avatar_url || '/avatar-placeholder.png'}
                  alt={doctor.user?.full_name}
                  className="w-12 h-12 rounded-full object-cover"
                />
                <div className="flex-1">
                  <h4 className="font-semibold text-gray-900">
                    BS. {doctor.user?.full_name}
                  </h4>
                  <p className="text-sm text-gray-600">{doctor.specialty}</p>
                  {doctor.years_of_experience && (
                    <p className="text-xs text-gray-500 mt-1">
                      {doctor.years_of_experience} năm kinh nghiệm
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
        {errors.doctor_id && (
          <p className="text-red-500 text-sm mt-2">{errors.doctor_id}</p>
        )}
      </div>

      {/* Step 2: Select Date & Time */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Calendar className="w-5 h-5 text-blue-600" />
          Bước 2: Chọn Ngày & Giờ
        </h3>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Ngày khám <span className="text-red-500">*</span>
            </label>
            <input
              type="date"
              value={formData.appointment_date}
              onChange={(e) => handleChange('appointment_date', e.target.value)}
              min={minDate}
              max={maxDate}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            {errors.appointment_date && (
              <p className="text-red-500 text-sm mt-1">{errors.appointment_date}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Giờ khám <span className="text-red-500">*</span>
            </label>
            <select
              value={formData.timeslot_id || ''}
              onChange={(e) => {
                const val = e.target.value;
                handleChange('timeslot_id', val ? Number(val) : '');
                // also set appointment_time for display
                const found = availableSlots.find(s => String(s.id) === String(val));
                if (found) handleChange('appointment_time', found.label);
                else handleChange('appointment_time', '');
              }}
              disabled={availableSlots.length === 0}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">{availableSlots.length ? 'Chọn khung giờ' : 'Không có khung giờ'}</option>
              {availableSlots.map(slot => (
                <option key={slot.id} value={slot.id}>{slot.label}</option>
              ))}
            </select>
            {availableSlots.length === 0 && (
              <p className="text-sm text-gray-500 mt-2">Bác sĩ không có lịch làm việc ngày đã chọn.</p>
            )}
            {errors.appointment_time && (
              <p className="text-red-500 text-sm mt-1">{errors.appointment_time}</p>
            )}
          </div>
        </div>
      </div>

      {/* Step 3: Reason & Notes */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-600" />
          Bước 3: Thông tin khám
        </h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Lý do khám <span className="text-red-500">*</span>
            </label>
            <textarea
              value={formData.reason}
              onChange={(e) => handleChange('reason', e.target.value)}
              rows={4}
              placeholder="Mô tả triệu chứng, lý do khám..."
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            {errors.reason && (
              <p className="text-red-500 text-sm mt-1">{errors.reason}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Ghi chú (tùy chọn)
            </label>
            <textarea
              value={formData.notes}
              onChange={(e) => handleChange('notes', e.target.value)}
              rows={3}
              placeholder="Thông tin bổ sung..."
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>
      </div>

      {/* Info Alert */}
      <Alert
        type="info"
        message="Lưu ý: Lịch hẹn sẽ được xác nhận trong vòng 24h. Bạn sẽ nhận được thông báo qua email và SMS."
      />

      {/* Submit Button */}
      <div className="flex justify-end gap-3">
        <Button
          type="submit"
          disabled={loading || creatingPatient}
          className="px-6"
        >
          {loading || creatingPatient ? 'Đang xử lý...' : 'Đặt lịch khám'}
        </Button>
      </div>
      {showRegisterRedirect && (
        <div className="mt-3 text-sm">
          <p className="text-red-600 mb-2">Bạn không có quyền tạo bệnh nhân từ đây.</p>
          <button onClick={() => navigate('/receptionist/register-patient')} className="px-3 py-2 bg-blue-600 text-white rounded">Mở trang đăng ký bệnh nhân</button>
        </div>
      )}
    </form>
  );
}