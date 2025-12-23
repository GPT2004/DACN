/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react';
import DashboardLayout from '../../components/layout/DashboardLayout';
import BookingForm from '../../components/common/BookingForm';
import { getAllDoctorsPublic, getSpecialties } from '../../services/doctorService';
import { useLocation } from 'react-router-dom';

export default function BookAppointmentPage() {
  const location = useLocation();
  const [specialties, setSpecialties] = useState([]); // array of objects {id,name,slug,...}
  const [doctors, setDoctors] = useState([]);
  const [selectedSpecialty, setSelectedSpecialty] = useState('');
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [success, setSuccess] = useState('');
  const [step, setStep] = useState(1); // 1: specialty, 2: doctor list, 3: booking form

  useEffect(() => {
    const fetchDoctors = async () => {
      try {
        // If a specialty is selected, fetch doctors filtered by specialty
        const params = selectedSpecialty ? { specialty: selectedSpecialty } : {};
        const result = await getAllDoctorsPublic(params);
        const doctorList = result?.data?.doctors || result?.data || result || [];
        setDoctors(Array.isArray(doctorList) ? doctorList : []);
      } catch (err) {
        console.error('Lỗi khi tải danh sách bác sĩ:', err);
        setDoctors([]);
      } finally {
        setLoading(false);
      }
    };
    fetchDoctors();
  }, []);

  useEffect(() => {
    // load specialties once
    const loadSpecialties = async () => {
      try {
        const res = await getSpecialties();
        const list = res?.data?.specialties || res?.data || [];
        setSpecialties(Array.isArray(list) ? list : []);
      } catch (err) {
        console.warn('Không thể tải chuyên khoa:', err);
        setSpecialties([]);
      }
    };
    loadSpecialties();
  }, []);

  useEffect(() => {
    // If query string contains doctorId, preselect and jump to form
    const q = new URLSearchParams(location.search);
    const doctorId = q.get('doctorId');
    if (doctorId) {
      // try to find doctor in current list first, otherwise fetch without filter
      const found = doctors.find((d) => String(d.id) === String(doctorId));
      if (found) {
        setSelectedDoctor(found);
        setShowForm(true);
        setStep(3);
      } else {
        // fetch all doctors and find
        (async () => {
          try {
            const res = await getAllDoctorsPublic();
            const all = res?.data?.doctors || res?.data || [];
            const arr = Array.isArray(all) ? all : [];
            const f = arr.find((d) => String(d.id) === String(doctorId));
            if (f) {
              setSelectedDoctor(f);
              setShowForm(true);
              setStep(3);
            }
          } catch (err) {
            // ignore
          }
        })();
      }
    }
  }, [location.search, doctors]);

  const handleBookingSuccess = (appointment) => {
    setShowForm(false);
    setSelectedDoctor(null);
    setSuccess('Đặt lịch thành công! Bác sĩ sẽ xác nhận sớm.');
    setTimeout(() => setSuccess(''), 5000);
  };

  if (loading) {
    return (
      <DashboardLayout>
        <div className="text-center py-8">Đang tải...</div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout role="patient">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Đặt lịch khám</h1>

        {success && (
          <div className="mb-4 p-4 bg-green-50 border border-green-200 text-green-700 rounded">
            {success}
          </div>
        )}

        <div>
          {/* Step navigation */}
          <div className="mb-6 flex items-center gap-4">
            <button
              className={`px-3 py-1 rounded ${step === 1 ? 'bg-emerald-600 text-white' : 'bg-gray-100'}`}
              onClick={() => setStep(1)}
            >
              1. Chọn chuyên khoa
            </button>
            <button
              className={`px-3 py-1 rounded ${step === 2 ? 'bg-emerald-600 text-white' : 'bg-gray-100'}`}
              onClick={() => setStep(2)}
              disabled={!selectedSpecialty}
            >
              2. Chọn bác sĩ
            </button>
            <button
              className={`px-3 py-1 rounded ${step === 3 ? 'bg-emerald-600 text-white' : 'bg-gray-100'}`}
              onClick={() => setStep(3)}
              disabled={!selectedDoctor}
            >
              3. Chọn khung giờ & hồ sơ
            </button>
          </div>

          {/* Step 1: specialties */}
          {step === 1 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Chọn chuyên khoa</h2>
                  {specialties.length === 0 ? (
                    <div className="text-gray-500">Không có chuyên khoa</div>
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                      {specialties.map((s) => (
                        <button
                          key={s.slug || s.id}
                          onClick={() => {
                            // store specialty as display name (string) for backend queries
                            const specialtyName = s.name || s;
                            setSelectedSpecialty(specialtyName);
                            setStep(2);
                            setLoading(true);
                            // fetch doctors filtered by this specialty (send name to match doctors.specialties values)
                            getAllDoctorsPublic({ specialty: specialtyName }).then((res) => {
                              const list = res?.data?.doctors || res?.data || [];
                              setDoctors(Array.isArray(list) ? list : []);
                              setLoading(false);
                            }).catch(() => setLoading(false));
                          }}
                          className={`p-3 text-left rounded border hover:shadow ${selectedSpecialty === (s.name || s) ? 'border-emerald-600 bg-emerald-50' : 'bg-white'}`}
                        >
                          <div className="font-semibold">{s.name || s}</div>
                        </button>
                      ))}
                    </div>
                  )}
            </div>
          )}

          {/* Step 2: doctors */}
          {step === 2 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Chọn bác sĩ - {specialties.find(sp => (sp.slug||sp.name) === selectedSpecialty)?.name || selectedSpecialty}</h2>
              {loading ? (
                <div className="text-gray-500">Đang tải...</div>
              ) : doctors.length === 0 ? (
                <div className="text-gray-500">Không có bác sĩ cho chuyên khoa này</div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {doctors.map((doctor) => (
                    <div
                      key={doctor.id}
                      className="bg-white rounded-lg shadow p-4 hover:shadow-lg transition cursor-pointer"
                      onClick={() => {
                        setSelectedDoctor(doctor);
                        setShowForm(true);
                        setStep(3);
                      }}
                    >
                      <h3 className="font-bold text-lg">{doctor.user?.full_name || 'N/A'}</h3>
                      <p className="text-sm text-gray-600 mb-2">
                        {doctor.specialties?.join(', ') || 'Chuyên khoa không rõ'}
                      </p>
                      <p className="text-sm text-gray-500 mb-3">{doctor.bio || 'Chưa có mô tả'}</p>
                      <p className="font-semibold text-blue-600">
                        {doctor.consultation_fee || 0} VNĐ
                      </p>
                      <button className="mt-3 w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600">
                        Đặt lịch
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Step 3: booking form */}
          {step === 3 && selectedDoctor && (
            <div className="max-w-2xl mx-auto">
              <BookingForm
                doctorId={selectedDoctor?.id}
                onSuccess={handleBookingSuccess}
                onCancel={() => {
                  setShowForm(false);
                  setSelectedDoctor(null);
                  setStep(2);
                }}
              />
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
