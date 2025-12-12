import React, {useEffect, useState} from 'react';
import { getDoctorSchedule, getTodayAppointments } from '../../services/doctorService';
export default function DoctorDashboardPage(){
  const [schedule,setSchedule]=useState([]);
  const [today,setToday]=useState([]);
  useEffect(()=>{ (async ()=>{ const s = await getDoctorSchedule(); if(s && s.data) setSchedule(s.data); const t = await getTodayAppointments(); if(t && t.data) setToday(t.data); })(); },[]);
  return (<div className='p-6'><h1 className='text-2xl font-bold mb-4'>Bảng điều khiển bác sĩ</h1><div className='grid grid-cols-2 gap-4'><div className='bg-white p-4 rounded shadow'><h2 className='font-semibold mb-2'>Lịch</h2>{schedule.length? schedule.map(x=>(<div key={x.id} className='p-2 border-b'>{x.date} {x.time}</div>)):'Không có lịch'}</div><div className='bg-white p-4 rounded shadow'><h2 className='font-semibold mb-2'>Lịch hẹn khám bệnh</h2>{today.length? today.map(a=>(<div key={a.id} className='p-2 border-b'>{a.patientName} - {a.time}</div>)):'Không có lịch hẹn khám bệnh'}</div></div></div>);
}
