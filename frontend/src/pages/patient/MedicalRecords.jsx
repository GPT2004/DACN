import React, {useEffect, useState} from 'react';
import { getPrescriptionsByPatient } from '../../services/prescriptionService';
export default function MedicalRecords(){
  const [pres,setPres]=useState([]);
  useEffect(()=>{ (async ()=>{ const r = await getPrescriptionsByPatient(1); if(r && r.data) setPres(r.data); })(); },[]);
  return (<div className='p-6'><h1 className='text-2xl font-bold mb-4'>Hồ sơ bệnh án</h1><div className='bg-white p-4 rounded shadow'>{pres.length? pres.map(p=>(<div key={p.id} className='border p-2 mb-2'>{p.title||'Đơn thuốc'}</div>)):'Không có dữ liệu'}</div></div>);
}
