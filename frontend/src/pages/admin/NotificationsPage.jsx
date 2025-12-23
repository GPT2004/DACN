/* eslint-disable no-unused-vars */
import React, {useEffect, useState} from 'react';
import PageWrapper from './_AdminWrapper';
export default function NotificationsPage(){
  const [notes,setNotes]=useState([]);
  useEffect(()=>{ /* TODO fetch notifications */ setNotes([]); },[]);
  return (<div className="p-6"><h1 className="text-2xl font-bold mb-4">Thông báo hệ thống</h1><div className="bg-white p-4 rounded shadow">{notes.length? notes.map(n=>(<div key={n.id} className='border p-2 mb-2'>{n.title}</div>)):'Không có thông báo'}</div></div>);
}
