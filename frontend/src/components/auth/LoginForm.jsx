/* eslint-disable no-console */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Mail, Lock, Eye, EyeOff, Loader2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import Input from '../common/Input';
import Button from '../common/Button';
import Alert from '../common/Alert';

export default function LoginForm() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [rememberMe, setRememberMe] = useState(false);

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setError('');
  };

  const handleSubmit = async (e) => {
  e.preventDefault();
  setError('');
  setLoading(true);

  try {
    const result = await login(formData.email, formData.password);
    console.log("LOGIN RESULT:", result);

    const user = result?.user;
    console.log("USER AFTER LOGIN:", user);

    if (!result.success || !user) {
      setError(result.error || "Đăng nhập thất bại");
      return;
    }

    const role = user?.role?.name || user?.role;

console.log("ROLE DETECTED RAW:", user.role);
console.log("ROLE FINAL:", role);

let redirectPath = "/";

switch (role?.toLowerCase()) {
  case "admin":
    redirectPath = "/admin";
    break;
  case "doctor":
    redirectPath = "/doctor";
    break;
  case "patient":
    redirectPath = "/patient";
    break;
  case "staff":
  case "receptionist":
    redirectPath = "/staff";
    break;
  default:
    redirectPath = "/";
    break;
}

console.log("REDIRECT TO:", redirectPath);

navigate(redirectPath, { replace: true });


  } catch (err) {
    console.error("LOGIN ERROR:", err);
    setError("Lỗi đăng nhập");
  } finally {
    setLoading(false);
  }
};





  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {error && (
        <Alert type="error" message={error} closable onClose={() => setError('')} />
      )}

      <Input
        label="Email"
        type="email"
        value={formData.email}
        onChange={(e) => handleChange('email', e.target.value)}
        leftIcon={Mail}
        placeholder="email@example.com"
        required
        disabled={loading}
      />

      <div className="relative">
        <Input
          label="Mật khẩu"
          type={showPassword ? 'text' : 'password'}
          value={formData.password}
          onChange={(e) => handleChange('password', e.target.value)}
          leftIcon={Lock}
          placeholder="Nhập mật khẩu"
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

      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={rememberMe}
            onChange={(e) => setRememberMe(e.target.checked)}
            className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
          />
          <span className="text-sm text-gray-700">Ghi nhớ đăng nhập</span>
        </label>

        <Link
          to="/forgot-password"
          className="text-sm text-blue-600 hover:text-blue-700"
        >
          Quên mật khẩu?
        </Link>
      </div>

      <Button
        type="submit"
        variant="primary"
        fullWidth
        loading={loading}
        disabled={loading}
      >
        {loading ? (
          <>
            <Loader2 className="animate-spin mr-2" size={18} />
            Đang đăng nhập...
          </>
        ) : (
          'Đăng nhập'
        )}
      </Button>

      <div className="text-center text-sm text-gray-600">
        Chưa có tài khoản?{' '}
        <Link to="/register" className="text-blue-600 hover:text-blue-700 font-medium">
          Đăng ký ngay
        </Link>
      </div>
    </form>
  );
}
