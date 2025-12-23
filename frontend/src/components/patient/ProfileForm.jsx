/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
// frontend/src/components/patient/ProfileForm.jsx
import React, { useState, useEffect } from 'react';
import { User, Mail, Phone, Calendar, MapPin, Heart, AlertCircle, Save } from 'lucide-react';
import { patientService } from '../../services/patientService';
import { useAuth } from '../../context/AuthContext';
import Button from '../common/Button';
import Input from '../common/Input';
import Select from '../common/Select';
import { GENDER_OPTIONS, BLOOD_TYPE_OPTIONS } from '../../utils/constants';
import { formatDate } from '../../utils/helpers';

export default function ProfileForm() {
  const { user, refreshUser } = useAuth();
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [profile, setProfile] = useState(null);
  
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    phone: '',
    dob: '',
    gender: 'Male',
    address: '',
    blood_type: '',
    height: '',
    weight: '',
    allergies: '',
    medical_history: '',
    emergency_contact: '',
  });

  const [errors, setErrors] = useState({});

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      setLoading(true);
      const response = await patientService.getMyProfile();
      const data = response.data;
      
      setProfile(data);
      setFormData({
        full_name: data.user?.full_name || '',
        email: data.user?.email || '',
        phone: data.user?.phone || '',
        dob: data.user?.dob || '',
        gender: data.user?.gender || 'Male',
        address: data.user?.address || '',
        blood_type: data.blood_type || '',
        height: data.height || '',
        weight: data.weight || '',
        allergies: data.allergies || '',
        medical_history: data.medical_history || '',
        emergency_contact: data.emergency_contact || '',
      });
    } catch (error) {
      console.error('Error fetching profile:', error);
      alert('Không thể tải thông tin hồ sơ');
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

  const validateForm = () => {
    const newErrors = {};

    if (!formData.full_name?.trim()) {
      newErrors.full_name = 'Vui lòng nhập họ tên';
    }

    if (!formData.phone?.trim()) {
      newErrors.phone = 'Vui lòng nhập số điện thoại';
    } else if (!/^(0|\+84)[0-9]{9,10}$/.test(formData.phone)) {
      newErrors.phone = 'Số điện thoại không hợp lệ';
    }

    if (!formData.email?.trim()) {
      newErrors.email = 'Vui lòng nhập email';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email không hợp lệ';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;

    try {
      setLoading(true);
      await patientService.updateMyProfile(formData);
      await refreshUser();
      alert('Cập nhật hồ sơ thành công!');
      setEditing(false);
      fetchProfile();
    } catch (error) {
      console.error('Error updating profile:', error);
      alert(error.response?.data?.message || 'Có lỗi xảy ra. Vui lòng thử lại!');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setEditing(false);
    setErrors({});
    // Reset form to original data
    if (profile) {
      setFormData({
        full_name: profile.user?.full_name || '',
        email: profile.user?.email || '',
        phone: profile.user?.phone || '',
        dob: profile.user?.dob || '',
        gender: profile.user?.gender || 'Male',
        address: profile.user?.address || '',
        blood_type: profile.blood_type || '',
        height: profile.height || '',
        weight: profile.weight || '',
        allergies: profile.allergies || '',
        medical_history: profile.medical_history || '',
        emergency_contact: profile.emergency_contact || '',
      });
    }
  };

  const calculateBMI = () => {
    if (formData.height && formData.weight) {
      const heightInMeters = formData.height / 100;
      const bmi = formData.weight / (heightInMeters * heightInMeters);
      return bmi.toFixed(1);
    }
    return null;
  };

  const getBMICategory = (bmi) => {
    if (!bmi) return '';
    const value = parseFloat(bmi);
    if (value < 18.5) return 'Thiếu cân';
    if (value < 25) return 'Bình thường';
    if (value < 30) return 'Thừa cân';
    return 'Béo phì';
  };

  const bmi = calculateBMI();

  if (loading && !profile) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-6 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-20 h-20 rounded-full bg-white text-blue-600 flex items-center justify-center text-2xl font-bold">
                {formData.full_name?.charAt(0) || 'U'}
              </div>
              <div>
                <h2 className="text-2xl font-bold">{formData.full_name || 'Người dùng'}</h2>
                <p className="text-blue-100 mt-1">Mã BN: {profile?.id || 'N/A'}</p>
              </div>
            </div>
            {!editing && (
              <Button
                type="button"
                onClick={() => setEditing(true)}
                variant="outline"
                className="bg-white text-blue-600 hover:bg-blue-50"
              >
                Chỉnh sửa
              </Button>
            )}
          </div>
        </div>

        <div className="p-6 space-y-8">
          {/* Personal Information */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <User className="w-5 h-5 mr-2 text-blue-600" />
              Thông tin cá nhân
            </h3>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Họ và tên <span className="text-red-500">*</span>
                </label>
                <Input
                  value={formData.full_name}
                  onChange={(e) => handleChange('full_name', e.target.value)}
                  disabled={!editing}
                  error={errors.full_name}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Email <span className="text-red-500">*</span>
                </label>
                <Input
                  type="email"
                  value={formData.email}
                  onChange={(e) => handleChange('email', e.target.value)}
                  disabled={!editing}
                  leftIcon={Mail}
                  error={errors.email}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Số điện thoại <span className="text-red-500">*</span>
                </label>
                <Input
                  value={formData.phone}
                  onChange={(e) => handleChange('phone', e.target.value)}
                  disabled={!editing}
                  leftIcon={Phone}
                  error={errors.phone}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Ngày sinh
                </label>
                <Input
                  type="date"
                  value={formData.dob}
                  onChange={(e) => handleChange('dob', e.target.value)}
                  disabled={!editing}
                  leftIcon={Calendar}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Giới tính
                </label>
                <Select
                  value={formData.gender}
                  onChange={(e) => handleChange('gender', e.target.value)}
                  options={GENDER_OPTIONS}
                  disabled={!editing}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Nhóm máu
                </label>
                <Select
                  value={formData.blood_type}
                  onChange={(e) => handleChange('blood_type', e.target.value)}
                  options={[
                    { value: '', label: 'Chưa xác định' },
                    ...BLOOD_TYPE_OPTIONS
                  ]}
                  disabled={!editing}
                />
              </div>

              <div className="col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Địa chỉ
                </label>
                <Input
                  value={formData.address}
                  onChange={(e) => handleChange('address', e.target.value)}
                  disabled={!editing}
                  leftIcon={MapPin}
                  placeholder="Nhập địa chỉ đầy đủ"
                />
              </div>
            </div>
          </div>

          {/* Health Information */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Heart className="w-5 h-5 mr-2 text-red-600" />
              Thông tin sức khỏe
            </h3>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Chiều cao (cm)
                </label>
                <Input
                  type="number"
                  value={formData.height}
                  onChange={(e) => handleChange('height', e.target.value)}
                  disabled={!editing}
                  placeholder="170"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Cân nặng (kg)
                </label>
                <Input
                  type="number"
                  value={formData.weight}
                  onChange={(e) => handleChange('weight', e.target.value)}
                  disabled={!editing}
                  placeholder="65"
                />
              </div>

              {bmi && (
                <div className="col-span-2 p-4 bg-blue-50 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">Chỉ số BMI:</span>
                    <div className="text-right">
                      <span className="text-xl font-bold text-blue-600">{bmi}</span>
                      <span className="ml-2 text-sm text-gray-600">
                        ({getBMICategory(bmi)})
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div className="col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Dị ứng
                </label>
                <textarea
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
                  rows="3"
                  value={formData.allergies}
                  onChange={(e) => handleChange('allergies', e.target.value)}
                  disabled={!editing}
                  placeholder="Liệt kê các dị ứng (thuốc, thức ăn, ...)"
                />
              </div>

              <div className="col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tiền sử bệnh
                </label>
                <textarea
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
                  rows="4"
                  value={formData.medical_history}
                  onChange={(e) => handleChange('medical_history', e.target.value)}
                  disabled={!editing}
                  placeholder="Các bệnh đã từng mắc, phẫu thuật, ..."
                />
              </div>
            </div>
          </div>

          {/* Emergency Contact */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <AlertCircle className="w-5 h-5 mr-2 text-orange-600" />
              Liên hệ khẩn cấp
            </h3>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Thông tin người thân (tên, mối quan hệ, số điện thoại)
              </label>
              <Input
                value={formData.emergency_contact}
                onChange={(e) => handleChange('emergency_contact', e.target.value)}
                disabled={!editing}
                placeholder="VD: Nguyễn Văn A (Vợ/Chồng) - 0987654321"
              />
            </div>
          </div>
        </div>

        {/* Actions */}
        {editing && (
          <div className="flex gap-3 p-6 border-t bg-gray-50">
            <Button
              type="button"
              onClick={handleCancel}
              variant="outline"
              className="flex-1"
              disabled={loading}
            >
              Hủy
            </Button>
            <Button
              type="submit"
              className="flex-1"
              disabled={loading}
            >
              <Save className="w-4 h-4 mr-2" />
              {loading ? 'Đang lưu...' : 'Lưu thay đổi'}
            </Button>
          </div>
        )}
      </div>
    </form>
  );
}