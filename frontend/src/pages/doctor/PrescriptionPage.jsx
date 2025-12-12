import React, { useState, useEffect } from 'react';
import PrescriptionForm from '../../components/doctor/PrescriptionForm';
import { prescriptionService } from '../../services/prescriptionService';
import { medicalRecordService } from '../../services/medicalRecordService';

export default function PrescriptionPage(){ 
	const [patientId, setPatientId] = useState('');
	const [medicalRecordId, setMedicalRecordId] = useState('');
	const [saving, setSaving] = useState(false);
	const [loading, setLoading] = useState(false);

	useEffect(() => {
		// read patientId, medical_record_id, appointment_id from query string if present
		try {
			const params = new URLSearchParams(window.location.search);
			const pid = params.get('patientId') || params.get('patient_id');
			const mrid = params.get('medical_record_id') || params.get('medicalRecordId');
			const apptId = params.get('appointment_id') || params.get('appointmentId');
			
			if (pid) setPatientId(pid);
			if (mrid) setMedicalRecordId(mrid);
			
			// If appointment_id provided but no medical_record_id, fetch medical records for this appointment
			if (apptId && !mrid) {
				setLoading(true);
				medicalRecordService.getMedicalRecords({ q: '', limit: 100 })
					.then(res => {
						const records = res?.data?.records || [];
						// Find medical record for this appointment
						const apptRecord = records.find(r => r.appointment_id === Number(apptId));
						if (apptRecord) {
							setMedicalRecordId(apptRecord.id);
						}
					})
					.finally(() => setLoading(false));
			}
		} catch (e) { 
			// Error parsing query params
		}
	}, []);

	const handleSave = async (data) => {
		if (!patientId) return alert('Vui lòng nhập ID bệnh nhân');
		try {
			setSaving(true);
			// transform medicines to expected API items structure
			const items = data.medicines.map(m => ({
				medicine_id: m.medicine_id || undefined,
				medicine_name: m.name,
				quantity: Number(m.quantity) || 1,
				unit_price: m.unit_price || 0,
				instructions: m.instructions || '',
				dosage: m.dosage || ''
			}));

			const payload = {
				patient_id: Number(patientId),
				medical_record_id: medicalRecordId ? Number(medicalRecordId) : undefined,
				items,
			};

			const res = await prescriptionService.createPrescription(payload);
			const created = res?.data?.data || res?.data || res;
			alert('✅ Lưu đơn thuốc thành công');

			// Ask whether to send to reception for invoicing
			const send = window.confirm('Gửi đơn thuốc cho lễ tân để tạo hóa đơn (kèm phí khám)?');
			if (send) {
				try {
					await prescriptionService.notifyReception(created.id || created);
					alert('✅ Đã gửi thông báo tới lễ tân.');
				} catch (e) {
					alert('Không thể gửi thông báo tới lễ tân: ' + (e?.response?.data?.message || e.message || ''));
				}
			}

			// reset form / patient id for next
			setPatientId('');
		} catch (err) {
			alert('Lỗi lưu đơn thuốc: ' + (err?.response?.data?.message || err.message || ''));
		} finally {
			setSaving(false);
		}
	};

	return (
		<div className='p-6'>
			<h1 className='text-2xl font-bold mb-4'>Kê đơn thuốc</h1>
			{loading ? (
				<div className='bg-white p-6 rounded shadow text-center'>Đang tải thông tin...</div>
			) : (
				<>
					<div className='bg-white p-4 rounded shadow mb-4 space-y-3'>
						<div>
							<label className='block text-sm font-medium mb-2'>ID bệnh nhân</label>
							<input type='number' value={patientId} onChange={e => setPatientId(e.target.value)} className='w-64 px-3 py-2 border rounded' />
						</div>
						{medicalRecordId && (
							<div>
								<label className='block text-sm font-medium text-gray-600'>Mã hồ sơ bệnh án</label>
								<div className='px-3 py-2 bg-gray-50 rounded border text-sm font-medium'>{medicalRecordId}</div>
							</div>
						)}
					</div>
					<div className='bg-white p-4 rounded shadow'>
						<PrescriptionForm patient={{ name: 'ID: ' + patientId }} onSave={handleSave} onCancel={() => {}} />
					</div>
				</>
			)}
		</div>
	); }
