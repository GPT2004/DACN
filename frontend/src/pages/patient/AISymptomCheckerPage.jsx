import React, {useState} from 'react';
import { checkSymptoms } from '../../services/aiService';
export default function AISymptomCheckerPage(){ const [text,setText]=useState(''); const [result,setResult]=useState(null);
const handle=async()=>{ const r = await checkSymptoms({text}); if(r && r.data) setResult(r.data); }
return (<div className='p-6'><h1 className='text-2xl font-bold mb-4'>AI Symptom Checker</h1><div className='bg-white p-4 rounded shadow'><textarea value={text} onChange={e=>setText(e.target.value)} className='w-full p-2 border rounded' rows={6}></textarea><button onClick={handle} className='mt-2 px-4 py-2 bg-blue-600 text-white rounded'>Check</button>{result && <pre className='mt-4 bg-gray-100 p-2 rounded'>{JSON.stringify(result,null,2)}</pre>}</div></div>); }
