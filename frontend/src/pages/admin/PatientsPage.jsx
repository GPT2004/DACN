import React, {useEffect, useState} from 'react';
import { getAllUsers } from '../../services/userService';
export default function PatientsPage(){
  const [patients,setPatients]=useState([]);
  useEffect(()=>{ (async ()=>{ const r = await getAllUsers({role:'PATIENT'}); if(r && r.data) setPatients(r.data); })(); },[]);
  return (<div className="p-6"><h1 className="text-2xl font-bold mb-4">Quản lý bệnh nhân</h1><div className="bg-white p-4 rounded shadow">{patients.length? (<table className='min-w-full'><thead><tr className='text-left text-sm text-gray-600'><th className='py-2'>#</th><th className='py-2'>Tên</th><th className='py-2'>Email</th></tr></thead><tbody>{patients.map((p,i)=>(<tr key={p.id} className='border-t'><td className='py-2'>{i+1}</td><td className='py-2'>{p.fullName||p.name}</td><td className='py-2'>{p.email}</td></tr>))}</tbody></table>) : 'Không có bệnh nhân'}</div></div>);
}
