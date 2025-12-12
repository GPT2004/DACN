import React, { useEffect, useState } from 'react';
import PublicHeader from '../../components/common/PublicHeader';
import PatientPageHeader from '../../components/patient/PatientPageHeader';
import Footer from '../../components/patient/clinic/Footer';
import { useAuth } from '../../context/AuthContext';
import { patientService } from '../../services/patientService';
import { createAppointment } from '../../services/appointmentService';
import { useNavigate, useLocation } from 'react-router-dom';
import { X } from 'lucide-react';
import ProfileUpdateForm from '../../components/patient/ProfileUpdateForm';
import ChangePasswordForm from '../../components/patient/ChangePasswordForm';

export default function ProfilePage() {
  const { user } = useAuth();
  const [patient, setPatient] = useState(null);
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const location = useLocation();
  const [showPendingBanner, setShowPendingBanner] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState(null);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    fetchPatientProfile();
    fetchMyPatients();
    // show banner if redirected from booking or pending appointment exists
    try {
      const q = new URLSearchParams(window.location.search);
      const returnUrl = q.get('returnUrl');
      const pending = sessionStorage.getItem('pendingAppointment');
      if (returnUrl || pending) setShowPendingBanner(true);
    } catch (e) {
      // ignore
    }
  }, []);

  const fetchPatientProfile = async () => {
    try {
      const response = await patientService.getMyProfile();
      const data = response?.data || response || {};
      setPatient(data);
      setFormData(data);
    } catch (err) {
      console.error('Lỗi khi tải hồ sơ:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchMyPatients = async () => {
    try {
      const resp = await patientService.getMyPatients();
      const data = resp?.data || resp || [];
      setPatients(data);
      // select the linked patient by default
      const linked = data.find(p => p.user && p.user.id === user?.id) || data[0] || null;
      setSelectedPatient(linked);
      if (linked) {
        // populate formData with selected
        setFormData(mapPatientToForm(linked));
      }
    } catch (err) {
      console.error('Lỗi khi tải danh sách hồ sơ:', err);
    }
  };

  const mapPatientToForm = (p) => ({
    full_name: p.full_name || p.user?.full_name || '',
    dob: p.dob ? new Date(p.dob).toISOString().split('T')[0] : (p.user?.dob ? p.user.dob.split('T')[0] : ''),
    phone: p.phone || p.user?.phone || '',
    gender: p.gender || p.user?.gender || '',
    occupation: p.occupation || '',
    id_type: p.id_type || '',
    id_number: p.id_number || '',
    email: p.email || p.user?.email || '',
    nationality: p.nationality || '',
    ethnicity: p.ethnicity || '',
    address: p.address || ''
  });

  // New handler when ProfileUpdateForm submits
  const handleUpdate = async (form) => {
    setSaving(true);
    try {
      // If we're in creating mode, call create endpoint
      if (creating) {
        const resp = await patientService.createDependentPatient(form);
        const created = resp?.data || resp || {};
        // refresh list and select created
        await fetchMyPatients();
        if (created && created.id) {
          setSelectedPatient(created);
          setFormData(mapPatientToForm(created));
        }
        setCreating(false);

        // resume pending appointment if exists (same as update flow)
        try {
          const pendingRaw = sessionStorage.getItem('pendingAppointment');
          if (pendingRaw) {
            const payload = JSON.parse(pendingRaw);
            try {
              await createAppointment(payload);
              sessionStorage.removeItem('pendingAppointment');
              alert('Tạo hồ sơ thành công! Đặt lịch đã được tạo.');
              window.location.href = '/patient/appointments';
              return;
            } catch (createErr) {
              console.error('Failed to auto-create appointment after profile create', createErr);
              alert('Tạo hồ sơ thành công, nhưng đặt lịch tự động thất bại: ' + (createErr.response?.data?.message || createErr.message));
            }
          }
        } catch (e) {
          console.warn('Error processing pending appointment', e);
        }

        alert('Tạo hồ sơ thành công!');
        return;
      }

      // Determine whether updating linked profile or dependent
      let response;
      if (selectedPatient && selectedPatient.user && selectedPatient.user.id === user?.id) {
        // updating own linked profile - use FormData (avatar etc.)
        const formDataToSend = new FormData();
        Object.keys(form).forEach((k) => {
          const v = form[k];
          if (v !== undefined && v !== null) formDataToSend.append(k, v);
        });
        response = await patientService.updateMyProfile(formDataToSend);
      } else if (selectedPatient) {
        // updating a dependent profile - call owner endpoint
        response = await patientService.updateDependentPatient(selectedPatient.id, form);
      } else {
        // fallback: update my profile
        const formDataToSend = new FormData();
        Object.keys(form).forEach((k) => {
          const v = form[k];
          if (v !== undefined && v !== null) formDataToSend.append(k, v);
        });
        response = await patientService.updateMyProfile(formDataToSend);
      }

      const updated = response?.data || response || {};
      // refresh lists
      await fetchMyPatients();
      // if updated is a patient object, set selected
      if (updated && updated.id) {
        setSelectedPatient(updated);
        setPatient(updated);
      } else {
        // otherwise reload main profile
        await fetchPatientProfile();
      }

      // resume pending appointment if exists
      try {
        const pendingRaw = sessionStorage.getItem('pendingAppointment');
        if (pendingRaw) {
          const payload = JSON.parse(pendingRaw);
          try {
            await createAppointment(payload);
            sessionStorage.removeItem('pendingAppointment');
            alert('Cập nhật hồ sơ thành công! Đặt lịch đã được tạo.');
            window.location.href = '/patient/appointments';
            return;
          } catch (createErr) {
            console.error('Failed to auto-create appointment after profile update', createErr);
            alert('Cập nhật hồ sơ thành công, nhưng đặt lịch tự động thất bại: ' + (createErr.response?.data?.message || createErr.message));
          }
        }
      } catch (e) {
        console.warn('Error processing pending appointment', e);
      }

      alert('Cập nhật hồ sơ thành công!');
    } catch (err) {
      console.error('Lỗi cập nhật hồ sơ:', err);
      alert(`Lỗi: ${err.response?.data?.message || err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedPatient) {
      alert('Không có hồ sơ được chọn để xóa');
      return;
    }

    if (!window.confirm('Bạn có chắc muốn xóa hồ sơ này? Hành động này không thể hoàn tác.')) return;

    setSaving(true);
    try {
      let response;
      // if the selected patient is linked to current user -> delete own profile
      if (selectedPatient.user && selectedPatient.user.id === user?.id) {
        response = await patientService.deleteMyProfile();
        alert('Hồ sơ của bạn đã được xóa.');
        // redirect to homepage or login
        window.location.href = '/';
        return;
      }

      // otherwise delete dependent
      response = await patientService.deleteDependentPatient(selectedPatient.id);
      alert('Hồ sơ phụ đã được xóa.');
      // refresh list
      await fetchMyPatients();
      setSelectedPatient(null);
      setFormData({});
    } catch (err) {
      console.error('Error deleting profile', err);
      alert('Xóa hồ sơ thất bại: ' + (err.response?.data?.message || err.message));
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    if (patient) {
      setFormData({
        full_name: patient.user?.full_name || '',
        phone: patient.user?.phone || '',
        gender: patient.gender || '',
        dob: patient.user?.dob ? patient.user.dob.split('T')[0] : '',
        address: patient.address || '',
      });
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col min-h-screen">
        <PublicHeader />
        <PatientPageHeader title="👤 Hồ sơ cá nhân" description="Thông tin cá nhân, thay đổi một mật khẩu, cập nhật thông tin tin." />
        <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Đang tải hồ sơ...</p>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen">
      <PublicHeader />
      <PatientPageHeader title="👤 Hồ sơ cá nhân" description="Thông tin cá nhân, thay đổi một mật khẩu, cập nhật thông tin tin." />
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">

        {showPendingBanner && (
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 text-blue-800 rounded">
            <strong>Vui lòng hoàn thiện hồ sơ</strong> — bạn đã bắt đầu đặt lịch; hãy hoàn thành hồ sơ để hệ thống tự động tạo lịch cho bạn sau khi lưu.
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <aside className="md:col-span-1">
            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-semibold">Hồ sơ của tôi</h4>
                <button className="text-sm text-blue-600" onClick={() => { setSelectedPatient(null); setFormData({}); setCreating(true); }}>+ Tạo mới</button>
              </div>
              <div className="divide-y">
                {patients.map(p => (
                    <div key={p.id} className={`py-3 cursor-pointer ${selectedPatient && selectedPatient.id === p.id ? 'bg-blue-50' : ''}`} onClick={() => { setSelectedPatient(p); setFormData(mapPatientToForm(p)); setCreating(false); }}>
                    <div className="font-medium">{p.full_name || p.user?.full_name || '—'}</div>
                    <div className="text-xs text-gray-500">{p.phone || p.user?.phone || ''}</div>
                  </div>
                ))}
              </div>
            </div>
          </aside>

          <section className="md:col-span-3">
            <div className="bg-white rounded-lg shadow-md p-8">
              <ProfileUpdateForm initial={formData || {}} onSubmit={handleUpdate} submitting={saving} onDelete={handleDelete} isCreate={creating} onCancel={() => { setCreating(false); if (patient) setFormData(mapPatientToForm(patient)); else setFormData({}); }} />
              <ChangePasswordForm />
            </div>
          </section>
        </div>
      </main>
      <Footer />
    </div>
  );
}

