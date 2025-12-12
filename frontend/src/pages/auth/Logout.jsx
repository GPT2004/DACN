import React, {useEffect} from 'react';
import { useNavigate } from 'react-router-dom';
export default function Logout(){ const nav = useNavigate(); useEffect(()=>{ localStorage.removeItem('token'); setTimeout(()=>nav('/login'),300); },[]); return (<div className='p-6'>Đang đăng xuất...</div>); }
