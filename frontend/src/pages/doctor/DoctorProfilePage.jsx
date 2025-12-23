/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-console */
import React, { useEffect, useState, useRef } from 'react';
import ImageViewer from '../../components/common/ImageViewer';
import DoctorLayout from '../../components/doctor/DoctorLayout';
import { useAuth } from '../../context/AuthContext';
import { getDoctorById, updateOwnProfile, getDoctorByUser } from '../../services/doctorService';

export default function DoctorProfilePage() {
  const { user, setUser } = useAuth();
  const [doctor, setDoctor] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [avatarPreview, setAvatarPreview] = useState('');
  const [avatarFile, setAvatarFile] = useState(null);
  const [showImageViewer, setShowImageViewer] = useState(false);
  const fileInputRef = useRef(null);

  const doctorId = user?.doctors?.[0]?.id || user?.doctor_id || null;

  useEffect(() => {
    (async () => {
      let resolvedDoctorId = doctorId;

      if (!resolvedDoctorId && user?.id) {
        try {
          const r = await getDoctorByUser(user.id);
          const docRes = r?.data || r || {};
          const first = Array.isArray(docRes) ? docRes[0] : (docRes?.doctors?.[0] || docRes);
          resolvedDoctorId = first?.id || null;
        } catch (err) {
          console.warn('could not resolve doctor by user id', err);
        }
      }

      if (!resolvedDoctorId) {
        setLoading(false);
        return;
      }

      try {
        const res = await getDoctorById(resolvedDoctorId);
        const data = res?.data || res || {};
        setDoctor(data);
        // set avatar preview from loaded data
        setAvatarPreview(data.user?.avatar_url || data.avatar_url || '');
      } catch (err) {
        console.error('load doctor', err);
      } finally {
        setLoading(false);
      }
    })();
  }, [doctorId]);

  const handleSave = async () => {
    if (!doctor) return;
    setSaving(true);
    try {
      // If an avatar file was selected, send multipart/form-data
      let res;
      if (avatarFile) {
        const fd = new FormData();
        fd.append('avatar_url', avatarFile);
        fd.append('full_name', doctor.user?.full_name || '');
        fd.append('phone', doctor.user?.phone || '');
        fd.append('dob', doctor.user?.dob || null);
        fd.append('bio', doctor.bio || '');
        const specs = typeof doctor.specialties === 'string' ? doctor.specialties : (Array.isArray(doctor.specialties) ? doctor.specialties.join(', ') : '');
        fd.append('specialties', specs);
        fd.append('gender', doctor.gender || '');
        fd.append('address', doctor.address || '');

        res = await updateOwnProfile(fd);
      } else {
        const payload = {
          full_name: doctor.user?.full_name || '',
          phone: doctor.user?.phone || '',
          dob: doctor.user?.dob || null,
          bio: doctor.bio || '',
          specialties: typeof doctor.specialties === 'string' 
            ? doctor.specialties.split(',').map(s => s.trim()).filter(Boolean)
            : doctor.specialties,
          gender: doctor.gender || '',
          address: doctor.address || ''
        };

        res = await updateOwnProfile(payload);
      }
      const updated = res?.data || res || {};
      // update auth user if returned
      setUser(prev => ({ ...prev, ...updated.user }));
      setDoctor(updated);
      alert('Cập nhật hồ sơ thành công!');
    } catch (err) {
      console.error('save profile', err);
      alert('Lỗi khi cập nhật hồ sơ: ' + (err.message || 'Vui lòng thử lại'));
    } finally {
      setSaving(false);
    }
  };

  const handleFile = (e) => {
    const f = e.target.files && e.target.files[0];
    setAvatarFile(f || null);
    if (f) {
      try {
        if (avatarPreview && avatarPreview.startsWith && avatarPreview.startsWith('blob:')) {
          URL.revokeObjectURL(avatarPreview);
        }
      } catch (err) {}
      setAvatarPreview(URL.createObjectURL(f));
    }
  };

  const openImageViewer = () => { if (avatarPreview) setShowImageViewer(true); };

  // cleanup blob url on unmount / when preview changes
  useEffect(() => {
    return () => {
      try {
        if (avatarPreview && avatarPreview.startsWith && avatarPreview.startsWith('blob:')) {
          URL.revokeObjectURL(avatarPreview);
        }
      } catch (err) {}
    };
  }, [avatarPreview]);

  if (loading) return <DoctorLayout><div className="text-center py-8">Đang tải...</div></DoctorLayout>;

  return (
    <DoctorLayout>
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Hồ sơ bác sĩ</h1>
        

        {doctor && (
          <div className="bg-white p-4 rounded shadow space-y-3">
            <div className="flex items-center gap-4">
              <div className="w-24 h-24 bg-gray-100 rounded overflow-hidden cursor-pointer" onClick={openImageViewer}>
                {avatarPreview ? (
                  <img src={avatarPreview} alt="avatar" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-400">No</div>
                )}
              </div>
              <div>
                <input ref={fileInputRef} className="hidden" type="file" accept="image/*" onChange={handleFile} />
                <button type="button" onClick={() => fileInputRef.current && fileInputRef.current.click()} className="px-3 py-1 border rounded bg-white">Chọn ảnh</button>
              </div>
            </div>
            {/* Họ tên */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">Họ tên</label>
              <input 
                className="w-full border p-2 rounded" 
                value={doctor.user?.full_name || ''} 
                onChange={e => setDoctor(d => ({ ...d, user: { ...d.user, full_name: e.target.value } }))} 
              />
            </div>

            {/* SĐT */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">Số điện thoại</label>
              <input 
                className="w-full border p-2 rounded" 
                value={doctor.user?.phone || ''} 
                onChange={e => setDoctor(d => ({ ...d, user: { ...d.user, phone: e.target.value } }))}
                placeholder="0987654321"
              />
            </div>

            {/* Ngày sinh */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">Ngày sinh</label>
              <input 
                type="date"
                className="w-full border p-2 rounded" 
                value={doctor.user?.dob ? doctor.user.dob.split('T')[0] : ''} 
                onChange={e => setDoctor(d => ({ ...d, user: { ...d.user, dob: e.target.value } }))}
              />
            </div>

            {/* Giới tính */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">Giới tính</label>
              <select 
                className="w-full border p-2 rounded"
                value={doctor.gender || ''}
                onChange={e => setDoctor(d => ({ ...d, gender: e.target.value }))}
              >
                <option value="">-- Chọn giới tính --</option>
                <option value="MALE">Nam</option>
                <option value="FEMALE">Nữ</option>
                <option value="OTHER">Khác</option>
              </select>
            </div>

            {/* Địa chỉ */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">Địa chỉ</label>
              <input 
                className="w-full border p-2 rounded" 
                value={doctor.address || ''} 
                onChange={e => setDoctor(d => ({ ...d, address: e.target.value }))}
                placeholder="Nhập địa chỉ"
              />
            </div>

            {/* Chuyên môn */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">Chuyên môn (cách nhau bởi dấu phẩy)</label>
              <input 
                className="w-full border p-2 rounded" 
                value={Array.isArray(doctor.specialties) ? doctor.specialties.join(', ') : (doctor.specialties || '')} 
                onChange={e => setDoctor(d => ({ ...d, specialties: e.target.value }))}
                placeholder="Nội khoa, Tim mạch, Ngoại khoa"
              />
            </div>

            {/* Tiểu sử */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">Tiểu sử</label>
              <textarea 
                className="w-full border p-2 rounded h-24" 
                value={doctor.bio || ''} 
                onChange={e => setDoctor(d => ({ ...d, bio: e.target.value }))}
                placeholder="Nhập tiểu sử chuyên môn của bạn"
              />
            </div>

            <div className="flex gap-2 pt-4">
              <button 
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400" 
                onClick={handleSave} 
                disabled={saving}
              >
                {saving ? 'Đang lưu...' : 'Lưu thay đổi'}
              </button>
            </div>
          </div>
        )}
      </div>
      {showImageViewer && (
        <ImageViewer src={avatarPreview} alt={doctor?.user?.full_name || 'Avatar'} onClose={() => setShowImageViewer(false)} />
      )}
    </DoctorLayout>
  );
}
