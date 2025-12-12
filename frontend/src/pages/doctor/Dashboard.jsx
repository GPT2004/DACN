import React, {useEffect, useState} from 'react';
import { getDoctorSchedule } from '../../services/doctorService';

export default function DoctorDashboard(){
  const [slots, setSlots] = useState([]);
  useEffect(()=>{ (async ()=>{ const res = await getDoctorSchedule(); if(res && res.data) setSlots(res.data); else if(Array.isArray(res)) setSlots(res) })(); },[]);
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Doctor Dashboard</h1>
      <div className="bg-white rounded shadow p-4">
        <h2 className="font-semibold">Upcoming Appointments</h2>
        <ul className="mt-2 space-y-2">{slots.map(s=>(<li key={s.id} className="p-2 border rounded">{s.patientName||s.patient || 'Patient'} — {s.time||s.slotTime}</li>))}</ul>
      </div>
    </div>
  );
}
