/* eslint-disable no-console */
import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { updateDoctor } from '../../services/doctorService';
import api from '../../services/api';
import ImageViewer from '../common/ImageViewer';

export default function EditDoctorModal({ doctor, onClose, onSuccess }) {
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    phone: '',
    specialty: '',
    license_number: '',
  });
  const [avatarFile, setAvatarFile] = useState(null);
  const [avatarPreview, setAvatarPreview] = useState('');
  const [showImageViewer, setShowImageViewer] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [specialties, setSpecialties] = useState([]);

  useEffect(() => {
    fetchSpecialties();
  }, []);

  useEffect(() => {
    if (doctor) {
      console.log('Doctor object:', doctor);
      setFormData({
        full_name: doctor.user?.full_name || doctor.full_name || doctor.fullName || doctor.name || '',
        email: doctor.user?.email || doctor.email || '',
        phone: doctor.user?.phone || doctor.phone || '',
        specialty: doctor.specialty || '',
        license_number: doctor.license_number || '',
      });
      // Set avatar preview if available
      const avatarUrl = doctor.user?.avatar_url || doctor.user?.avatarUrl || doctor.avatar_url || '';
      setAvatarPreview(avatarUrl);
    }
  }, [doctor]);

  useEffect(() => {
    return () => {
      if (avatarPreview && avatarPreview.startsWith('blob:')) {
        URL.revokeObjectURL(avatarPreview);
      }
    };
  }, [avatarPreview]);

  const openImageViewer = () => {
    if (avatarPreview) setShowImageViewer(true);
  };

  const closeImageViewer = () => setShowImageViewer(false);

  const fetchSpecialties = async () => {
    try {
      const response = await api.get('/doctors/public/specialties/list');
      console.log('Specialties response:', response);
      // Ensure we get an array
      const specialtiesData = Array.isArray(response) ? response : 
                             Array.isArray(response?.data) ? response.data :
                             Array.isArray(response?.specialties) ? response.specialties : [];
      setSpecialties(specialtiesData);
    } catch (error) {
      console.error('Error fetching specialties:', error);
      setSpecialties([]); // Fallback to empty array
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError('');
  };

  const handleFileChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      setAvatarFile(file);
      const preview = URL.createObjectURL(file);
      setAvatarPreview(preview);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.full_name.trim() || !formData.email.trim()) {
      setError('Vui lòng điền tên và email');
      return;
    }

    // Validate phone format (10-11 digits)
    if (formData.phone && !/^[0-9]{10,11}$/.test(formData.phone)) {
      setError('Số điện thoại phải có 10-11 chữ số');
      return;
    }

    try {
      setLoading(true);
      // Use doctor.id as the doctor ID for update
      const doctorId = doctor.id;
      if (!doctorId) {
        setError('Không tìm thấy ID bác sĩ');
        return;
      }
      console.log('Sending data:', formData);
      console.log('Doctor ID:', doctorId);

      // If avatarFile is present, construct FormData
      let payload = formData;
      if (avatarFile) {
        const fd = new FormData();
        Object.keys(formData).forEach(k => {
          const v = formData[k];
          if (v !== undefined && v !== null) fd.append(k, v);
        });
        // backend expects field name 'avatar_url'
        fd.append('avatar_url', avatarFile);
        payload = fd;
      }

      await updateDoctor(doctorId, payload);
      onSuccess();
      onClose();
    } catch (err) {
      console.error('Update error:', err);
      console.error('Error response:', err.response?.data);
      console.error('Error status:', err.response?.status);
      setError(err.response?.data?.message || err.response?.data?.error || 'Cập nhật thất bại');
    } finally {
      setLoading(false);
    }
  };

  if (!doctor) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 max-h-[600px] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 flex items-center justify-between p-6 border-b border-gray-200 bg-white">
          <h2 className="text-xl font-bold text-gray-900">Chỉnh sửa bác sĩ</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 transition"
          >
            <X size={24} />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
              {error}
            </div>
          )}

          {/* Avatar upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Ảnh đại diện</label>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-16 h-16 bg-gray-100 rounded overflow-hidden cursor-pointer" onClick={openImageViewer}>
                  {avatarPreview ? <img src={avatarPreview} alt="avatar" className="w-full h-full object-cover" /> : <div className="w-full h-full flex items-center justify-center text-gray-400">No</div>}
                </div>
              <div>
                <input type="file" accept="image/*" onChange={handleFileChange} />
              </div>
                  {showImageViewer && (
                    <ImageViewer src={avatarPreview} alt={formData.full_name || 'Avatar'} onClose={closeImageViewer} />
                  )}
            </div>
          </div>

          {/* Full Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Họ và tên
            </label>
            <input
              type="text"
              name="full_name"
              value={formData.full_name}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              placeholder="Nhập họ và tên"
            />
          </div>

          {/* Email */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Email
            </label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              placeholder="Nhập email"
            />
          </div>

          {/* Phone */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Số điện thoại
            </label>
            <input
              type="tel"
              name="phone"
              value={formData.phone}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              placeholder="Nhập số điện thoại"
            />
          </div>

          {/* Specialty */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Chuyên khoa
            </label>
            <select
              name="specialty"
              value={formData.specialty}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
            >
              <option value="">Chọn chuyên khoa</option>
              {Array.isArray(specialties) && specialties.map(spec => (
                <option key={spec.id || spec.name} value={spec.name}>{spec.name}</option>
              ))}
            </select>
          </div>

          {/* License Number */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Số giấy phép
            </label>
            <input
              type="text"
              name="license_number"
              value={formData.license_number}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              placeholder="Nhập số giấy phép hành nghề"
            />
          </div>

          {/* Buttons */}
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition font-medium"
            >
              Hủy
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition font-medium"
            >
              {loading ? 'Đang lưu...' : 'Lưu'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
