import React, { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import api from '../../services/api';

export default function ConfirmRegistration() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');
  const [status, setStatus] = useState('loading');
  const [message, setMessage] = useState('');

  useEffect(() => {
    const confirm = async () => {
      if (!token) {
        setStatus('error');
        setMessage('Token không tồn tại');
        return;
      }
      try {
        const res = await api.get(`/auth/confirm-registration?token=${encodeURIComponent(token)}`);
        if (res && (res.success || res.user)) {
          setStatus('success');
          setMessage('Xác nhận đăng ký thành công. Bạn có thể đăng nhập ngay bây giờ.');
        } else {
          setStatus('error');
          setMessage(res && (res.message || res.error) ? (res.message || res.error) : 'Xác nhận thất bại');
        }
      } catch (err) {
        setStatus('error');
        setMessage(err && err.message ? err.message : 'Lỗi kết nối');
      }
    };

    confirm();
  }, [token]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <div className="w-full max-w-md bg-white p-8 rounded-lg shadow-lg text-center">
        {status === 'loading' && <p>Đang xác nhận...</p>}
        {status === 'success' && (
          <>
            <h2 className="text-xl font-semibold">Xác nhận thành công</h2>
            <p className="mt-2 text-gray-600">{message}</p>
            <div className="mt-4">
              <Link to="/login" className="text-blue-600 font-medium">Đăng nhập</Link>
            </div>
          </>
        )}
        {status === 'error' && (
          <>
            <h2 className="text-xl font-semibold text-red-600">Xác nhận thất bại</h2>
            <p className="mt-2 text-gray-600">{message}</p>
            <div className="mt-4">
              <Link to="/register" className="text-blue-600 font-medium">Quay lại đăng ký</Link>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
