import React, {useEffect, useState} from 'react';
import { getLabResults } from '../../services/labService';
export default function LabResultsPage(){ const [results,setResults]=useState([]); useEffect(()=>{ (async ()=>{ const r = await getLabResults(1); if(r && r.data) setResults(r.data); })(); },[]); return (<div className='p-6'><h1 className='text-2xl font-bold mb-4'>Kết quả xét nghiệm</h1><div className='bg-white p-4 rounded shadow'>{results.length? results.map(res=>(<div key={res.id} className='py-2 border-b'>{res.testName}: {res.value}</div>)):'Không có kết quả'}</div></div>); }
