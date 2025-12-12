import React, { useState, useEffect, useRef } from 'react';
import ImageViewer from '../common/ImageViewer';
import { COUNTRIES } from '../../data/countries';

export default function ProfileUpdateForm({ initial = {}, onSubmit, submitting, onDelete, isCreate = false, onCancel }) {
  const [form, setForm] = useState({
    full_name: initial.full_name || initial.user?.full_name || '',
    dob: initial.dob ? new Date(initial.dob).toISOString().split('T')[0] : '',
    phone: initial.phone || initial.user?.phone || '',
    gender: initial.gender || initial.user?.gender || '',
    occupation: initial.occupation || '',
    id_type: initial.id_type || '',
    id_number: initial.id_number || '',
    email: initial.email || initial.user?.email || '',
    nationality: initial.nationality || '',
    ethnicity: initial.ethnicity || '',
    address: initial.address || '',
    avatar_url: null,
  });

  const handleChange = (key) => (e) => setForm(prev => ({ ...prev, [key]: e.target.value }));

  useEffect(() => {
    setForm({
      full_name: initial.full_name || initial.user?.full_name || '',
      dob: initial.dob ? new Date(initial.dob).toISOString().split('T')[0] : (initial.user?.dob ? initial.user.dob.split('T')[0] : ''),
      phone: initial.phone || initial.user?.phone || '',
      gender: initial.gender || initial.user?.gender || '',
      occupation: initial.occupation || '',
      id_type: initial.id_type || '',
      id_number: initial.id_number || '',
      email: initial.email || initial.user?.email || '',
      nationality: initial.nationality || '',
      ethnicity: initial.ethnicity || '',
      address: initial.address || '',
      avatar_url: initial.user?.avatar_url || initial.avatar_url || null,
    });
  }, [initial]);

  const [avatarPreview, setAvatarPreview] = useState('');
  const [showImageViewer, setShowImageViewer] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (initial.user?.avatar_url || initial.avatar_url) {
      setAvatarPreview(initial.user?.avatar_url || initial.avatar_url);
    }
  }, [initial]);

  const handleFile = (e) => {
    const f = e.target.files && e.target.files[0];
    setForm(prev => ({ ...prev, avatar_url: f }));
    if (f) {
      // revoke previous blob url to avoid memory leak
      try {
        if (avatarPreview && avatarPreview.startsWith && avatarPreview.startsWith('blob:')) {
          URL.revokeObjectURL(avatarPreview);
        }
      } catch (err) {
        // ignore
      }
      setAvatarPreview(URL.createObjectURL(f));
    }
  };

  // cleanup any created blob URL on unmount
  useEffect(() => {
    return () => {
      try {
        if (avatarPreview && avatarPreview.startsWith && avatarPreview.startsWith('blob:')) {
          URL.revokeObjectURL(avatarPreview);
        }
      } catch (err) {
        // ignore
      }
    };
  }, [avatarPreview]);

  const openImageViewer = () => {
    if (avatarPreview) setShowImageViewer(true);
  };

  const closeImageViewer = () => setShowImageViewer(false);

  const submit = (e) => {
    e.preventDefault();
    if (onSubmit) onSubmit(form);
  };

  return (
    <form onSubmit={submit} className="bg-white p-6 rounded shadow">
      <h3 className="text-lg font-semibold mb-4">{isCreate ? 'Tạo hồ sơ mới' : 'Cập nhật thông tin'}</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium">Ảnh đại diện</label>
          <div className="flex items-center gap-3">
            <div className="w-20 h-20 bg-gray-100 rounded overflow-hidden cursor-pointer" onClick={openImageViewer}>
              {avatarPreview ? (
                <img src={avatarPreview} alt="avatar" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-400">No</div>
              )}
            </div>
            <div>
              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFile} className="hidden" />
              <button type="button" onClick={() => fileInputRef.current && fileInputRef.current.click()} className="px-3 py-1 border rounded bg-white">Chọn ảnh</button>
            </div>
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium">Họ và tên</label>
          <input value={form.full_name} onChange={handleChange('full_name')} className="w-full p-2 border rounded" />
        </div>

        <div>
          <label className="block text-sm font-medium">Ngày sinh</label>
          <input type="date" value={form.dob} onChange={handleChange('dob')} className="w-full p-2 border rounded" />
        </div>

        <div>
          <label className="block text-sm font-medium">Quốc tịch</label>
          <select value={form.nationality} onChange={handleChange('nationality')} className="w-full p-2 border rounded">
            <option value="">-- Chọn quốc gia --</option>
            {COUNTRIES.map(c => (
              <option key={c.code} value={c.name}>{c.name === 'Vietnam' ? 'Việt Nam' : c.name}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium">Số điện thoại</label>
          <input value={form.phone} onChange={handleChange('phone')} className="w-full p-2 border rounded" />
        </div>

        <div>
          <label className="block text-sm font-medium">Giới tính</label>
          <select value={form.gender} onChange={handleChange('gender')} className="w-full p-2 border rounded">
            <option value="">-- Chọn --</option>
            <option value="MALE">Nam</option>
            <option value="FEMALE">Nữ</option>
            <option value="OTHER">Khác</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium">Nghề nghiệp</label>
          <select value={form.occupation} onChange={handleChange('occupation')} className="w-full p-2 border rounded">
            <option value="">-- Chọn nghề nghiệp --</option>
            <option value="STUDENT">Học sinh/Sinh viên</option>
            <option value="EMPLOYEE">Nhân viên</option>
            <option value="SELF_EMPLOYED">Tự doanh</option>
            <option value="RETIRED">Hưu trí</option>
            <option value="UNEMPLOYED">Thất nghiệp</option>
            <option value="OTHER">Khác</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium">Loại giấy tờ (CCCD/CMND/Hộ chiếu)</label>
          <select value={form.id_type} onChange={handleChange('id_type')} className="w-full p-2 border rounded">
            <option value="">-- Chọn loại giấy tờ --</option>
            <option value="CCCD">CCCD</option>
            <option value="CMND">CMND</option>
            <option value="PASSPORT">Hộ chiếu</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium">Số giấy tờ</label>
          <input value={form.id_number} onChange={handleChange('id_number')} className="w-full p-2 border rounded" />
        </div>

        <div>
          <label className="block text-sm font-medium">Email</label>
          <input value={form.email} onChange={handleChange('email')} className="w-full p-2 border rounded" />
        </div>

        

        <div>
          <label className="block text-sm font-medium">Dân tộc</label>
          <input value={form.ethnicity} onChange={handleChange('ethnicity')} className="w-full p-2 border rounded" />
        </div>
      </div>

      <hr className="my-4" />

      <h4 className="font-medium mb-2">Địa chỉ</h4>
      <div className="mb-4">
        <textarea placeholder="Địa chỉ hiện tại" value={form.address} onChange={handleChange('address')} className="w-full p-2 border rounded" rows={3} />
      </div>

      <div className="flex gap-2 justify-end items-center">
        <button type="button" onClick={() => { if (onCancel) onCancel(); else onSubmit && onSubmit(null); }} className="px-4 py-2 bg-gray-200 rounded">Hủy</button>
        {typeof onDelete === 'function' && (
          <button type="button" onClick={() => {
            // ask for confirmation in the form before calling parent
            if (window.confirm('Bạn có chắc muốn xóa hồ sơ này? Hành động này không thể hoàn tác.')) {
              onDelete();
            }
          }} className="px-4 py-2 bg-red-600 text-white rounded">Xóa</button>
        )}
        <button type="submit" disabled={submitting} className="px-4 py-2 bg-blue-600 text-white rounded">{submitting ? (isCreate ? 'Đang tạo...' : 'Đang lưu...') : (isCreate ? 'Tạo mới' : 'Cập nhật')}</button>
      </div>
      {showImageViewer && (
        <ImageViewer src={avatarPreview} alt={form.full_name || 'Avatar'} onClose={closeImageViewer} />
      )}
    </form>
  );
}
