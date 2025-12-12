import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { User, Mail, Lock, Phone, Calendar, Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import Input from '../common/Input';
import Button from '../common/Button';
import Select from '../common/Select';
import Alert from '../common/Alert';
import { GENDER_OPTIONS } from '../../utils/constants';

export default function RegisterForm() {
  const navigate = useNavigate();
  const { register } = useAuth();
  
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    phone: '',
    password: '',
    confirm_password: '',
    dob: '',
    gender: 'Male',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [errors, setErrors] = useState({});

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
    setError('');
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
    } else if (!/^(0|\+84)[0-9]{9,10}$/.test(formData.phone)) {
      newErrors.phone = 'Số điện thoại không hợp lệ';
    }

    if (!formData.password) {
      newErrors.password = 'Vui lòng nhập mật khẩu';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Mật khẩu tối thiểu 6 ký tự';
    }

    if (!formData.confirm_password) {
      newErrors.confirm_password = 'Vui lòng xác nhận mật khẩu';
    } else if (formData.password !== formData.confirm_password) {
      newErrors.confirm_password = 'Mật khẩu xác nhận không khớp';
    }

    if (!formData.dob) {
      newErrors.dob = 'Vui lòng chọn ngày sinh';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;

    setLoading(true);
    setError('');

    try {
      const { confirm_password, ...registerData } = formData;
      const result = await register(registerData);
      
      if (result.success) {
        navigate('/patient');
      } else {
        setError(result.error || 'Đăng ký thất bại');
      }
    } catch (err) {
      setError('Có lỗi xảy ra. Vui lòng thử lại');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <Alert type="error" message={error} closable onClose={() => setError('')} />
      )}

      <Input
        label="Họ và tên"
        value={formData.full_name}
        onChange={(e) => handleChange('full_name', e.target.value)}
        leftIcon={User}
        placeholder="Nguyễn Văn A"
        error={errors.full_name}
        required
        disabled={loading}
      />

      <Input
        label="Email"
        type="email"
        value={formData.email}
        onChange={(e) => handleChange('email', e.target.value)}
        leftIcon={Mail}
        placeholder="email@example.com"
        error={errors.email}
        required
        disabled={loading}
      />

      <Input
        label="Số điện thoại"
        value={formData.phone}
        onChange={(e) => handleChange('phone', e.target.value)}
        leftIcon={Phone}
        placeholder="0123456789"
        error={errors.phone}
        required
        disabled={loading}
      />

      <div className="grid grid-cols-2 gap-4">
        <Input
          label="Ngày sinh"
          type="date"
          value={formData.dob}
          onChange={(e) => handleChange('dob', e.target.value)}
          leftIcon={Calendar}
          error={errors.dob}
          required
          disabled={loading}
        />

        <Select
          label="Giới tính"
          value={formData.gender}
          onChange={(e) => handleChange('gender', e.target.value)}
          options={GENDER_OPTIONS}
          required
          disabled={loading}
        />
      </div>

      <div className="relative">
        <Input
          label="Mật khẩu"
          type={showPassword ? 'text' : 'password'}
          value={formData.password}
          onChange={(e) => handleChange('password', e.target.value)}
          leftIcon={Lock}
          placeholder="Tối thiểu 6 ký tự"
          error={errors.password}
          required
          disabled={loading}
        />
        <button
          type="button"
          onClick={() => setShowPassword(!showPassword)}
          className="absolute right-3 top-[38px] text-gray-400 hover:text-gray-600"
          tabIndex={-1}
        >
          {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>
      </div>

      <div className="relative">
        <Input
          label="Xác nhận mật khẩu"
          type={showConfirmPassword ? 'text' : 'password'}
          value={formData.confirm_password}
          onChange={(e) => handleChange('confirm_password', e.target.value)}
          leftIcon={Lock}
          placeholder="Nhập lại mật khẩu"
          error={errors.confirm_password}
          required
          disabled={loading}
        />
        <button
          type="button"
          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
          className="absolute right-3 top-[38px] text-gray-400 hover:text-gray-600"
          tabIndex={-1}
        >
          {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>
      </div>

      <Button
        type="submit"
        variant="primary"
        fullWidth
        loading={loading}
        disabled={loading}
      >
        Đăng ký
      </Button>

      <div className="text-center text-sm text-gray-600">
        Đã có tài khoản?{' '}
        <Link to="/login" className="text-blue-600 hover:text-blue-700 font-medium">
          Đăng nhập
        </Link>
      </div>
    </form>
  );
}