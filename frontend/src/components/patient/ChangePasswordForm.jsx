import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';

export default function ChangePasswordForm() {
  const { changePassword } = useAuth();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setMessage(null);
    setError(null);

    if (!currentPassword || !newPassword) {
      setError('Vui lòng nhập đầy đủ mật khẩu hiện tại và mật khẩu mới');
      return;
    }
    if (newPassword !== confirmPassword) {
      setError('Mật khẩu mới và xác nhận không khớp');
      return;
    }
    setLoading(true);
    try {
      const res = await changePassword(currentPassword, newPassword);
      if (res && res.success === false) {
        setError(res.error || 'Không thể đổi mật khẩu');
      } else {
        setMessage('Đổi mật khẩu thành công');
        setCurrentPassword('');
        setNewPassword('');
        setConfirmPassword('');
      }
    } catch (err) {
      setError(err.message || 'Đổi mật khẩu thất bại');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 mt-6">
      <h3 className="text-lg font-semibold mb-3">Đổi mật khẩu</h3>
      <form onSubmit={submit}>
        <div className="mb-3">
          <label className="block text-sm font-medium">Mật khẩu hiện tại</label>
          <input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} className="w-full p-2 border rounded" />
        </div>

        <div className="mb-3">
          <label className="block text-sm font-medium">Mật khẩu mới</label>
          <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} className="w-full p-2 border rounded" />
        </div>

        <div className="mb-3">
          <label className="block text-sm font-medium">Xác nhận mật khẩu mới</label>
          <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="w-full p-2 border rounded" />
        </div>

        {error && <div className="text-sm text-red-600 mb-2">{error}</div>}
        {message && <div className="text-sm text-green-600 mb-2">{message}</div>}

        <div className="flex justify-end">
          <button type="submit" disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded">{loading ? 'Đang xử lý...' : 'Đổi mật khẩu'}</button>
        </div>
      </form>
    </div>
  );
}
