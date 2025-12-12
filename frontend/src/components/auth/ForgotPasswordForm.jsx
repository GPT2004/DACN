import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Mail, ArrowLeft, CheckCircle } from 'lucide-react';
import { authService } from '../../services/authService';
import Input from '../common/Input';
import Button from '../common/Button';
import Alert from '../common/Alert';

export default function ForgotPasswordForm() {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!email.trim()) {
      setError('Vui lòng nhập email');
      return;
    }

    if (!/\S+@\S+\.\S+/.test(email)) {
      setError('Email không hợp lệ');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // TODO: Implement forgot password API
      // await authService.forgotPassword(email);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      setSuccess(true);
    } catch (err) {
      setError('Có lỗi xảy ra. Vui lòng thử lại');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="text-center space-y-6">
        <div className="flex justify-center">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
            <CheckCircle className="w-10 h-10 text-green-600" />
          </div>
        </div>
        
        <div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            Kiểm tra email của bạn
          </h3>
          <p className="text-gray-600">
            Chúng tôi đã gửi hướng dẫn đặt lại mật khẩu đến email:
          </p>
          <p className="font-medium text-gray-900 mt-1">{email}</p>
        </div>

        <div className="text-sm text-gray-600">
          Không nhận được email?{' '}
          <button
            onClick={() => setSuccess(false)}
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            Gửi lại
          </button>
        </div>

        <Link
          to="/login"
          className="inline-flex items-center text-blue-600 hover:text-blue-700"
        >
          <ArrowLeft size={16} className="mr-1" />
          Quay lại đăng nhập
        </Link>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="text-center mb-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-2">
          Quên mật khẩu?
        </h3>
        <p className="text-gray-600">
          Nhập email của bạn để nhận hướng dẫn đặt lại mật khẩu
        </p>
      </div>

      {error && (
        <Alert type="error" message={error} closable onClose={() => setError('')} />
      )}

      <Input
        label="Email"
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        leftIcon={Mail}
        placeholder="email@example.com"
        required
        disabled={loading}
      />

      <Button
        type="submit"
        variant="primary"
        fullWidth
        loading={loading}
        disabled={loading}
      >
        Gửi hướng dẫn
      </Button>

      <div className="text-center">
        <Link
          to="/login"
          className="inline-flex items-center text-sm text-gray-600 hover:text-gray-900"
        >
          <ArrowLeft size={16} className="mr-1" />
          Quay lại đăng nhập
        </Link>
      </div>
    </form>
  );
}