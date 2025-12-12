import React, {useEffect, useState} from 'react';
import { getDoctorSchedule } from '../../services/doctorService';
export default function PatientsList(){
  const [patients,setPatients]=useState([]);
  useEffect(()=>{ (async ()=>{ const r=await getDoctorSchedule(); if(r && r.data) setPatients(r.data.map(s=>({id:s.id, name:s.patientName}))); else if(Array.isArray(r)) setPatients(r.map(s=>({id:s.id,name:s.patientName}))); })(); },[]);
  return (<div className='p-6'><h1 className='text-2xl font-bold mb-4'>Danh sách bệnh nhân</h1><div className='bg-white p-4 rounded shadow'><ul>{patients.map(p=>(<li key={p.id} className='py-2 border-b'>{p.name}</li>))}</ul></div></div>)
}
