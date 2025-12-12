import React, {useEffect, useState} from 'react';
import { getDoctorSchedule } from '../../services/doctorService';
export default function SchedulePage(){
  const [slots,setSlots]=useState([]);
  useEffect(()=>{ (async ()=>{ const r=await getDoctorSchedule(); if(r && r.data) setSlots(r.data); else if(Array.isArray(r)) setSlots(r); })(); },[]);
  return (<div className='p-6'><h1 className='text-2xl font-bold mb-4'>Lịch khám</h1><div className='bg-white p-4 rounded shadow'><ul className='space-y-2'>{slots.map(s=>(<li key={s.id} className='p-2 border rounded'>{s.date} {s.time} - {s.patientName}</li>))}</ul></div></div>);
}
