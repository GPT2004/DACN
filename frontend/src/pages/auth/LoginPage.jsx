import React, {useState} from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

export default function LoginPage(){
  const [email,setEmail]=useState('');
  const [password,setPassword]=useState('');
  const [loading,setLoading]=useState(false);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { login: ctxLogin } = useAuth();

  // Map role to dashboard path
  const getRolePath = (role) => {
    if (!role) return '/';
    
    const rolePathMap = {
      'Admin': '/admin',
      'Doctor': '/doctor',
      'Receptionist': '/receptionist',
      'Reception': '/receptionist',  // Backend uses "Reception" instead of "Receptionist"
      'Pharmacist': '/pharmacist',
      'LabTech': '/labtech',
      'Patient': '/patient',
      // Lowercase alternatives
      'admin': '/admin',
      'doctor': '/doctor',
      'receptionist': '/receptionist',
      'reception': '/receptionist',
      'pharmacist': '/pharmacist',
      'labtech': '/labtech',
      'patient': '/patient'
    };
    
    return rolePathMap[role] || '/';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try{
      // Use AuthContext login so context and storage are updated in one place
      const res = await ctxLogin(email, password);
      if (res && res.success && res.user) {
        // Kiểm tra returnUrl từ query params
        const returnUrl = searchParams.get('returnUrl');
        
        if (returnUrl) {
          // Redirect về trang trước đó
          navigate(returnUrl);
        } else {
          // Điều hướng theo role (handle both string and object shapes)
          let userRole = 'patient';
          if (res.user) {
            if (typeof res.user.role === 'string') userRole = res.user.role;
            else if (typeof res.user.role === 'object') userRole = res.user.role.name || res.user.role.role || 'patient';
          }
          const rolePath = getRolePath(userRole);
          navigate(rolePath);
        }
      } else {
        alert(res.error || res.message || 'Login failed');
      }
    }catch(e){
      alert('Network error');
    }finally{ setLoading(false); }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <form onSubmit={handleSubmit} className="w-full max-w-md bg-white p-8 rounded-lg shadow-lg">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold">Đăng nhập</h1>
          <p className="text-sm text-gray-500 mt-1">Đăng nhập để truy cập hồ sơ và lịch hẹn của bạn</p>
        </div>

        <label className="block mb-4">
          <div className="text-sm font-medium text-gray-700">Email</div>
          <input type="email" value={email} onChange={e=>setEmail(e.target.value)} required className="mt-1 p-3 border rounded w-full focus:outline-none focus:ring-2 focus:ring-blue-200"/>
        </label>

        <label className="block mb-4">
          <div className="text-sm font-medium text-gray-700">Mật khẩu</div>
          <input type="password" value={password} onChange={e=>setPassword(e.target.value)} required className="mt-1 p-3 border rounded w-full focus:outline-none focus:ring-2 focus:ring-blue-200"/>
        </label>

        <div className="flex items-center justify-between mt-6">
          <button type="submit" disabled={loading} className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-60">{loading? 'Đang...' : 'Đăng nhập'}</button>
        </div>

        <div className="mt-4 flex items-center justify-between text-sm">
          <Link to="/forgot-password" className="text-blue-600">Quên mật khẩu?</Link>
          <div className="text-gray-600">Chưa có tài khoản? <Link to="/register" className="text-blue-600 font-medium">Đăng ký</Link></div>
        </div>
      </form>
    </div>
  );
}
