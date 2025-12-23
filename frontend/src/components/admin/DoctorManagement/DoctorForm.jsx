/* eslint-disable no-console */
// frontend/src/components/admin/DoctorManagement/DoctorForm.jsx
import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { doctorService } from '../../../services/doctorService';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Modal from '../../common/Modal';
import axios from 'axios';

export default function DoctorForm({ isOpen, onClose, doctor, onSuccess }) {
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    phone: '',
    password: '',
    dob: '',
    gender: 'Male',
    specialty: '',
    license_number: '',
    years_of_experience: '',
    education: '',
    bio: '',
    consultation_fee: '',
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [specialties, setSpecialties] = useState([]);

  useEffect(() => {
    fetchSpecialties();
  }, [isOpen]);

  useEffect(() => {
    if (doctor) {
      setFormData({
        full_name: doctor.user?.full_name || '',
        email: doctor.user?.email || '',
        phone: doctor.user?.phone || '',
        password: '',
        dob: doctor.user?.dob || '',
        gender: doctor.user?.gender || 'Male',
        specialty: doctor.specialty || '',
        license_number: doctor.license_number || '',
        years_of_experience: doctor.years_of_experience || '',
        education: doctor.education || '',
        bio: doctor.bio || '',
        consultation_fee: doctor.consultation_fee || '',
      });
    } else {
      resetForm();
    }
  }, [doctor]);

  const fetchSpecialties = async () => {
    try {
      const response = await axios.get('/api/doctors/public/specialties/list');
      setSpecialties(response.data || []);
    } catch (error) {
      console.error('Error fetching specialties:', error);
    }
  };

  const resetForm = () => {
    setFormData({
      full_name: '',
      email: '',
      phone: '',
      password: '',
      dob: '',
      gender: 'Male',
      specialty: '',
      license_number: '',
      years_of_experience: '',
      education: '',
      bio: '',
      consultation_fee: '',
    });
    setErrors({});
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.full_name.trim()) {
      newErrors.full_name = 'Vui lòng nhập họ tên';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Vui lòng nhập email';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email không hợp lệ';
    }

    if (!formData.phone.trim()) {
      newErrors.phone = 'Vui lòng nhập số điện thoại';
    }

    if (!doctor && !formData.password) {
      newErrors.password = 'Vui lòng nhập mật khẩu';
    }

    if (!formData.specialty) {
      newErrors.specialty = 'Vui lòng chọn chuyên khoa';
    }

    if (!formData.license_number.trim()) {
      newErrors.license_number = 'Vui lòng nhập số chứng chỉ hành nghề';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    try {
      setLoading(true);

      if (doctor) {
        await doctorService.updateDoctor(doctor.id, formData);
      } else {
        await doctorService.createDoctor(formData);
      }

      onSuccess();
      onClose();
      resetForm();
    } catch (error) {
      console.error('Submit error:', error);
      alert('Có lỗi xảy ra. Vui lòng thử lại!');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-800">
            {doctor ? 'Chỉnh sửa Bác sĩ' : 'Thêm Bác sĩ mới'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          <div className="grid grid-cols-2 gap-6">
            {/* Thông tin cá nhân */}
            <div className="col-span-2">
              <h3 className="text-lg font-semibold mb-4">Thông tin cá nhân</h3>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Họ tên <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.full_name}
                onChange={(e) => handleChange('full_name', e.target.value)}
                placeholder="Nhập họ tên"
              />
              {errors.full_name && (
                <p className="text-red-500 text-sm mt-1">{errors.full_name}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email <span className="text-red-500">*</span>
              </label>
              <Input
                type="email"
                value={formData.email}
                onChange={(e) => handleChange('email', e.target.value)}
                placeholder="email@example.com"
              />
              {errors.email && (
                <p className="text-red-500 text-sm mt-1">{errors.email}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Số điện thoại <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.phone}
                onChange={(e) => handleChange('phone', e.target.value)}
                placeholder="0123456789"
              />
              {errors.phone && (
                <p className="text-red-500 text-sm mt-1">{errors.phone}</p>
              )}
            </div>

            {!doctor && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Mật khẩu <span className="text-red-500">*</span>
                </label>
                <Input
                  type="password"
                  value={formData.password}
                  onChange={(e) => handleChange('password', e.target.value)}
                  placeholder="Nhập mật khẩu"
                />
                {errors.password && (
                  <p className="text-red-500 text-sm mt-1">{errors.password}</p>
                )}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Ngày sinh
              </label>
              <Input
                type="date"
                value={formData.dob}
                onChange={(e) => handleChange('dob', e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Giới tính
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                value={formData.gender}
                onChange={(e) => handleChange('gender', e.target.value)}
              >
                <option value="Male">Nam</option>
                <option value="Female">Nữ</option>
                <option value="Other">Khác</option>
              </select>
            </div>

            {/* Thông tin chuyên môn */}
            <div className="col-span-2 mt-4">
              <h3 className="text-lg font-semibold mb-4">Thông tin chuyên khoa</h3>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Chuyên khoa <span className="text-red-500">*</span>
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                value={formData.specialty}
                onChange={(e) => handleChange('specialty', e.target.value)}
              >
                <option value="">Chọn chuyên khoa</option>
                {specialties.map(spec => (
                  <option key={spec.id} value={spec.name}>{spec.name}</option>
                ))}
              </select>
              {errors.specialty && (
                <p className="text-red-500 text-sm mt-1">{errors.specialty}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Số chứng chỉ hành nghề <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.license_number}
                onChange={(e) => handleChange('license_number', e.target.value)}
                placeholder="Nhập số chứng chỉ"
              />
              {errors.license_number && (
                <p className="text-red-500 text-sm mt-1">{errors.license_number}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Số năm kinh nghiệm
              </label>
              <Input
                type="number"
                value={formData.years_of_experience}
                onChange={(e) => handleChange('years_of_experience', e.target.value)}
                placeholder="Số năm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Phí khám (VNĐ)
              </label>
              <Input
                type="number"
                value={formData.consultation_fee}
                onChange={(e) => handleChange('consultation_fee', e.target.value)}
                placeholder="200000"
              />
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Học vấn
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="3"
                value={formData.education}
                onChange={(e) => handleChange('education', e.target.value)}
                placeholder="Nhập thông tin học vấn, bằng cấp..."
              />
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Giới thiệu
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="4"
                value={formData.bio}
                onChange={(e) => handleChange('bio', e.target.value)}
                placeholder="Giới thiệu ngắn gọn về bác sĩ..."
              />
            </div>
          </div>
        </form>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t bg-gray-50">
          <Button onClick={onClose} disabled={loading}>
            Hủy
          </Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Đang xử lý...' : doctor ? 'Cập nhật' : 'Thêm mới'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}