import React, {useEffect, useState} from 'react';
import { getDoctorSchedule } from '../../services/doctorService';
export default function SchedulesPage(){
  const [schedules,setSchedules]=useState([]);
  useEffect(()=>{ (async ()=>{ const r = await getDoctorSchedule(); if(r && r.data) setSchedules(r.data); })(); },[]);
  return (<div className="p-6"><h1 className="text-2xl font-bold mb-4">Quản lý lịch</h1><div className="bg-white p-4 rounded shadow">{schedules.length? (<table className='min-w-full'><thead><tr className='text-left text-sm text-gray-600'><th className='py-2'>Bác sĩ</th><th className='py-2'>Ngày</th><th className='py-2'>Giờ</th></tr></thead><tbody>{schedules.map(s=>(<tr key={s.id} className='border-t'><td className='py-2'>{s.doctorName||s.doctor}</td><td className='py-2'>{s.date}</td><td className='py-2'>{s.time}</td></tr>))}</tbody></table>) : 'Không có lịch'}</div></div>);
}
