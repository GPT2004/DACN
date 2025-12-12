import React, { useEffect, useState } from 'react';
import { prescriptionService } from '../../services/prescriptionService';
import { invoiceService } from '../../services/invoiceService';
import CreateInvoiceFromPrescriptionModal from '../../components/receptionist/CreateInvoiceFromPrescriptionModal';

function getQueryParam(name) {
  try {
    const params = new URLSearchParams(window.location.search);
    return params.get(name);
  } catch (err) {
    return null;
  }
}

export default function PrescriptionsInbox(){
  const [list, setList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [modalPrescriptionId, setModalPrescriptionId] = useState(null);

  const fetchList = async () => {
    setLoading(true);
    try {
      const res = await prescriptionService.getForInvoicing({ page: 1, limit: 50 });
      const data = res?.data?.data || res?.data || res;
      setList(data.prescriptions || []);
      // If URL contains prescription_id param, prompt for invoice creation
      const pid = getQueryParam('prescription_id');
      if (pid) {
        const numeric = parseInt(pid);
        const found = (data.prescriptions || []).find(p => p.id === numeric);
        if (found) {
          // Open modal to create invoice for this prescription (no confirm)
          setTimeout(() => {
            setModalPrescriptionId(numeric);
            setShowCreateModal(true);
          }, 250);
        }
      }
    } catch (e) {
      console.error(e);
      alert('Không lấy được danh sách đơn');
    } finally { setLoading(false); }
  };

  useEffect(() => { fetchList(); }, []);

  const handleCreateInvoice = async (prescriptionId) => {
    try {
      const res = await invoiceService.createFromPrescription(prescriptionId);
      const invoice = res?.data?.data || res?.data || res;
      alert('Tạo hoá đơn thành công: ' + (invoice.id || invoice));
      fetchList();
    } catch (e) {
      alert('Không thể tạo hoá đơn: ' + (e?.response?.data?.message || e.message || ''));
    }
  };

  return (
    <div className='p-6'>
      <h1 className='text-2xl font-bold mb-4'>Đơn thuốc chờ lập hoá đơn</h1>
      {loading ? <div>Đang tải...</div> : (
        <div className='space-y-3'>
          {list.length === 0 && <div className='bg-white p-4 rounded shadow'>Không có đơn nào.</div>}
          {list.map(p => (
            <div key={p.id} className='bg-white p-4 rounded shadow flex justify-between items-start'>
              <div>
                <div className='font-medium'>Đơn #{p.id} — Bệnh nhân: {p.patient?.user?.full_name || 'N/A'}</div>
                <div className='text-sm text-gray-600'>Bác sĩ: {p.doctor?.user?.full_name || 'N/A'}</div>
                <div className='text-sm mt-2'>Tổng thuốc: {p.total_amount || 0} VND</div>
              </div>
              <div className='flex flex-col gap-2'>
                <button onClick={() => { setModalPrescriptionId(p.id); setShowCreateModal(true); }} className='px-3 py-1 bg-green-600 text-white rounded'>Tạo hoá đơn</button>
                <a className='px-3 py-1 border rounded text-sm' href={'/receptionist/invoices'}>Mở hoá đơn</a>
              </div>
            </div>
          ))}
        </div>
      )}
      {showCreateModal && (
        <div>
          <CreateInvoiceFromPrescriptionModal
            initialPrescriptionId={modalPrescriptionId}
            onClose={() => { setShowCreateModal(false); setModalPrescriptionId(null); fetchList(); }}
            onInvoiceCreated={(inv) => { alert('Tạo hoá đơn thành công'); setShowCreateModal(false); setModalPrescriptionId(null); fetchList(); }}
          />
        </div>
      )}
    </div>
  );
}
