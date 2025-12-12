import React, { useState } from 'react';
import { register as registerUser } from '../../services/authService';
import { useNavigate, Link } from 'react-router-dom';

export default function RegisterPage() {
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [phone, setPhone] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);
  const nav = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    if (password !== confirmPassword) {
      alert('Mật khẩu xác nhận không khớp');
      setLoading(false);
      return;
    }
    try {
      const payload = { email, full_name: fullName, phone, password, confirm_password: confirmPassword };
      const r = await registerUser(payload);
      if (r && (r.errCode === 0 || r.success || r.message)) {
        // Show check-your-email message instead of immediate login
        setSent(true);
      } else {
        alert('Đăng ký thất bại: ' + (r && (r.message || r.error) ? (r.message || r.error) : ''));
      }
    } catch (err) {
      alert('Network error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <form onSubmit={handleSubmit} className="w-full max-w-md bg-white p-8 rounded-lg shadow-lg">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold">Tạo tài khoản</h1>
          <p className="text-sm text-gray-500 mt-1">Đăng ký để quản lý lịch khám và hồ sơ bệnh án</p>
        </div>
        {!sent ? (
          <>
            <label className="block mb-4">
              <div className="text-sm font-medium text-gray-700">Họ và tên</div>
              <input value={fullName} onChange={e => setFullName(e.target.value)} placeholder="Họ và tên" required className="mt-1 p-3 border rounded w-full focus:outline-none focus:ring-2 focus:ring-blue-200" />
            </label>

            <label className="block mb-4">
              <div className="text-sm font-medium text-gray-700">Số điện thoại</div>
              <input value={phone} onChange={e => setPhone(e.target.value)} placeholder="Số điện thoại" required className="mt-1 p-3 border rounded w-full focus:outline-none focus:ring-2 focus:ring-blue-200" />
            </label>

            <label className="block mb-4">
              <div className="text-sm font-medium text-gray-700">Email</div>
              <input value={email} onChange={e => setEmail(e.target.value)} placeholder="Email" type="email" required className="mt-1 p-3 border rounded w-full focus:outline-none focus:ring-2 focus:ring-blue-200" />
            </label>

            <label className="block mb-4">
              <div className="text-sm font-medium text-gray-700">Mật khẩu</div>
              <input value={password} onChange={e => setPassword(e.target.value)} placeholder="Mật khẩu" type="password" required className="mt-1 p-3 border rounded w-full focus:outline-none focus:ring-2 focus:ring-blue-200" />
            </label>

            <label className="block mb-4">
              <div className="text-sm font-medium text-gray-700">Xác nhận mật khẩu</div>
              <input value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} placeholder="Xác nhận mật khẩu" type="password" required className="mt-1 p-3 border rounded w-full focus:outline-none focus:ring-2 focus:ring-blue-200" />
            </label>

            <button type="submit" disabled={loading} className="mt-2 w-full bg-blue-600 text-white py-3 rounded-md hover:bg-blue-700 disabled:opacity-60">{loading ? 'Đang...' : 'Đăng ký'}</button>

            <div className="mt-4 text-sm text-center text-gray-600">Bạn đã có tài khoản? <Link to="/login" className="text-blue-600 font-medium">Đăng nhập</Link></div>
          </>
        ) : (
          <div className="text-center">
            <h2 className="text-lg font-semibold">Kiểm tra email</h2>
            <p className="mt-2 text-sm text-gray-600">Chúng tôi đã gửi một email xác nhận tới địa chỉ của bạn. Vui lòng mở email và nhấp vào liên kết để hoàn tất đăng ký.</p>
            <div className="mt-4 text-sm text-gray-600">Sau khi xác nhận, bạn sẽ được chuyển đến trang đăng nhập.</div>
          </div>
        )}
      </form>
    </div>
  );
}
