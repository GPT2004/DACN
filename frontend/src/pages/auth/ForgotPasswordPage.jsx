/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import { forgotPassword, checkEmail } from '../../services/authService';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [exists, setExists] = useState(null);
  const [checking, setChecking] = useState(false);

  const handleCheck = async (value) => {
    if (!value || value.indexOf('@') === -1) {
      setExists(null);
      return;
    }
    setChecking(true);
    try {
      const res = await checkEmail(value);
      if (res && res.data && typeof res.data.data !== 'undefined') {
        setExists(!!res.data.data.exists);
      } else if (res && typeof res.exists !== 'undefined') {
        setExists(!!res.exists);
      } else if (res && typeof res.data === 'boolean') {
        setExists(res.data);
      } else {
        // fallback: if response shape unexpected, treat as unknown
        setExists(null);
      }
    } catch (err) {
      setExists(null);
    } finally {
      setChecking(false);
    }
  };

  const handle = async (e) => {
    e.preventDefault();
    // If we don't know whether the email exists yet, check now
    if (exists === null) {
      await handleCheck(email);
    }

    if (exists === false) {
      // better UX than alert: setExists to false and return
      alert('Email không tồn tại trong hệ thống');
      return;
    }
    try {
      const r = await forgotPassword(email);
      setSent(true);
    } catch (err) {
      alert('Gửi yêu cầu thất bại');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      {sent ? (
        <div className="bg-white p-6 rounded shadow">Yêu cầu đổi mật khẩu đã gửi. Vui lòng kiểm tra email.</div>
      ) : (
        <form onSubmit={handle} className="w-full max-w-md bg-white p-6 rounded shadow">
          <h1 className="text-xl font-semibold mb-4">Quên mật khẩu</h1>
          <input
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              setExists(null);
            }}
            onBlur={(e) => handleCheck(e.target.value)}
            placeholder="Email"
            className="mb-2 p-2 border rounded w-full"
          />
          {checking && <div className="text-sm text-gray-500 mb-2">Đang kiểm tra...</div>}
          {exists === false && <div className="text-sm text-red-600 mb-2">Email không tồn tại</div>}
          {exists === true && <div className="text-sm text-green-600 mb-2">Email hợp lệ</div>}
          <button
            type="submit"
            disabled={checking || !email || email.indexOf('@') === -1 || exists === false}
            className="mt-2 w-full bg-blue-600 text-white py-2 rounded"
          >
            Gửi
          </button>
        </form>
      )}
    </div>
  );
}
