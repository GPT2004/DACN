import React, {useState} from 'react';
export default function SettingsPage(){
  const [clinicName,setClinicName]=useState('My Clinic');
  const handleSave=()=>{ alert('Saved'); }
  return (<div className="p-6"><h1 className="text-2xl font-bold mb-4">Cấu hình hệ thống</h1><div className="bg-white p-4 rounded shadow"><label className="block mb-2">Tên phòng khám<input value={clinicName} onChange={e=>setClinicName(e.target.value)} className="mt-1 p-2 border rounded w-full"/></label><button onClick={handleSave} className="mt-2 px-4 py-2 bg-blue-600 text-white rounded">Lưu</button></div></div>);
}
