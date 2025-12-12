import React, { useState, useMemo } from 'react';
import { X, Search, FileText } from 'lucide-react';
import { prescriptionService } from '../../services/prescriptionService';
import { invoiceService } from '../../services/invoiceService';
import { medicineService } from '../../services/medicineService';

export default function CreateInvoiceFromPrescriptionModal({ initialPrescriptionId = null, onClose, onInvoiceCreated }) {
  const [prescriptionId, setPrescriptionId] = useState('');
  const [consultFee, setConsultFee] = useState(0);
  const [loading, setLoading] = useState(false);
  const [prescription, setPrescription] = useState(null);
  const [itemsState, setItemsState] = useState([]);
  const [error, setError] = useState('');

  const loadPrescription = async (id = null) => {
    const idToLoad = id || prescriptionId;
    if (!idToLoad) return setError('Vui lòng nhập ID đơn thuốc.');
    try {
      setError('');
      setLoading(true);
      // make API call
      const res = await prescriptionService.getPrescriptionById(idToLoad);
      // normalize response to actual prescription object (handle several wrapping shapes)
      let pres = res?.data || res;
      if (pres && pres.data) pres = pres.data;
      if (pres && pres.prescription) pres = pres.prescription;

      if (!pres || !pres.id) {
        throw new Error('Không tìm thấy đơn thuốc (response empty)');
      }

      setPrescription(pres);

      // Default consult fee from doctor or appointment if available
      const defaultConsult = pres?.doctor?.consultation_fee || pres?.appointment?.doctor?.consultation_fee || 0;
      setConsultFee(defaultConsult || 0);

      // Build items state: ensure unit_price populated (fetch medicine price if needed)
      const rawItems = typeof pres.items === 'string' ? JSON.parse(pres.items) : pres.items || [];
      const enriched = await Promise.all(rawItems.map(async (it) => {
        const base = {
          medicine_id: it.medicine_id || it.id || null,
          description: it.medicine_name || it.description || it.name || 'Thuốc',
          quantity: Number(it.quantity || it.qty || 1),
          unit_price: Number(it.unit_price || 0),
        };
        if ((!base.unit_price || base.unit_price === 0) && base.medicine_id) {
          try {
            const m = await medicineService.getMedicineById(base.medicine_id);
            const md = m.data || m;
            base.unit_price = Number(md.price || base.unit_price || 0);
          } catch (e) {
            // ignore, keep existing unit_price
            console.error('Failed to fetch medicine price', e?.message || e);
          }
        }
        base.amount = (Number(base.unit_price || 0) * Number(base.quantity || 0)) || 0;
        return base;
      }));
      setItemsState(enriched);
    } catch (err) {
      console.error('loadPrescription error:', err);
      setError(err?.response?.data?.message || err.message || 'Không tìm thấy đơn thuốc.');
      setPrescription(null);
      setItemsState([]);
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    if (initialPrescriptionId) {
      setPrescriptionId(String(initialPrescriptionId));
      // load directly using provided id (avoid stale state)
      setTimeout(() => loadPrescription(initialPrescriptionId), 50);
    }
  }, [initialPrescriptionId]);

  const handleCreateInvoice = async () => {
    if (!prescription) return setError('Vui lòng load đơn thuốc trước.');
    try {
      setLoading(true);
      // If this modal loaded an existing prescription, use the server endpoint
      // that creates invoice FROM the prescription so the prescription is marked
      // as invoiced and will disappear from the inbox. If no prescription loaded,
      // fall back to creating a custom invoice.
      let inv;
      if (prescription && prescription.id) {
        const created = await invoiceService.createFromPrescription(prescription.id);
        inv = created.data || created;
      } else {
        // Build items from current editable itemsState + consultation fee
        // normalize items as integers (Joi expects integers)
        const items = itemsState.map(it => {
          const qty = parseInt(it.quantity || 1, 10) || 0;
          const unit = parseInt(it.unit_price || 0, 10) || 0;
          return { type: 'medicine', description: it.description, quantity: qty, unit_price: unit, amount: unit * qty };
        });
        if (consultFee && Number(consultFee) > 0) {
          const fee = parseInt(consultFee || 0, 10) || 0;
          items.unshift({ type: 'consultation', description: 'Phí khám', quantity: 1, unit_price: fee, amount: fee });
        }

        const subtotal = items.reduce((s, it) => s + (it.amount || 0), 0);

        // ensure appointment_id and patient_id are integers (or undefined)
        const appointmentId = prescription?.appointment?.id ? parseInt(prescription.appointment.id, 10) : undefined;
        const patientId = prescription?.patient_id ? parseInt(prescription.patient_id, 10) : (prescription?.patient?.id ? parseInt(prescription.patient.id, 10) : undefined);

        const payload = {
          // only include appointment_id when it's a valid number
          ...(appointmentId ? { appointment_id: appointmentId } : {}),
          patient_id: patientId,
          items,
          subtotal,
          tax: 0,
          discount: 0,
          total: subtotal,
        };

        const created = await invoiceService.createInvoice(payload);
        inv = created.data || created;
      }

      onInvoiceCreated(inv);
      onClose();
    } catch (err) {
      console.error('createInvoice error:', err);
      const msg = err?.response?.data?.message || err?.message || 'Tạo hoá đơn thất bại';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const subtotal = useMemo(() => itemsState.reduce((s, it) => s + ((Number(it.unit_price || 0) * Number(it.quantity || 0)) || 0), 0), [itemsState]);
  const tax = 0;
  const discount = 0;
  const consult = Number(consultFee || 0);
  const total = subtotal + consult + tax - discount;

  const updateItem = (index, key, value) => {
    setItemsState(prev => {
      const copy = [...prev];
      copy[index] = { ...copy[index], [key]: key === 'description' ? value : Number(value) };
      copy[index].amount = (Number(copy[index].unit_price || 0) * Number(copy[index].quantity || 0));
      return copy;
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl">
        <div className="p-4 border-b flex items-center justify-between">
          <h3 className="text-lg font-semibold flex items-center gap-2"><FileText /> Tạo hóa đơn từ đơn thuốc</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800"><X /></button>
        </div>
        <div className="p-4 space-y-4">
          {error && <div className="p-3 bg-red-50 text-red-800 rounded">{error}</div>}
          <div className="grid grid-cols-3 gap-3 items-center">
            <input type="text" placeholder="ID đơn thuốc" value={prescriptionId} onChange={e => setPrescriptionId(e.target.value)} className="col-span-2 px-3 py-2 border rounded" />
            <button onClick={loadPrescription} className="px-4 py-2 bg-blue-600 text-white rounded">Load</button>
          </div>

          {prescription && (
            <div className="bg-gray-50 p-3 rounded space-y-2">
              <div className="flex justify-between text-sm"><span>Bệnh nhân</span><span>{prescription.patient?.user?.full_name || prescription.patient?.full_name}</span></div>
              <div className="flex justify-between text-sm"><span>Bác sĩ</span><span>{prescription.doctor?.user?.full_name}</span></div>
              <div>
                <p className="text-sm font-medium">Thuốc:</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left">
                        <th className="py-1">Mô tả</th>
                        <th className="py-1">Số lượng</th>
                        <th className="py-1">Đơn giá</th>
                        <th className="py-1">Thành tiền</th>
                      </tr>
                    </thead>
                    <tbody>
                      {itemsState.map((it, i) => (
                        <tr key={i} className="align-top">
                          <td className="py-1"><input value={it.description} onChange={e => updateItem(i, 'description', e.target.value)} className="px-2 py-1 border rounded w-full" /></td>
                          <td className="py-1"><input type="number" value={it.quantity} onChange={e => updateItem(i, 'quantity', e.target.value)} className="w-20 px-2 py-1 border rounded" /></td>
                          <td className="py-1"><input type="number" value={it.unit_price} onChange={e => updateItem(i, 'unit_price', e.target.value)} className="w-32 px-2 py-1 border rounded" /></td>
                          <td className="py-1">{(Number(it.amount) || 0).toLocaleString('vi-VN')} đ</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="pt-2 border-t">
                <div className="flex justify-between text-sm"><span>Thuốc</span><span>{subtotal.toLocaleString('vi-VN')} đ</span></div>
                <div className="flex justify-between text-sm"><span>Phí khám</span><span>{consult.toLocaleString('vi-VN')} đ</span></div>
                <div className="flex justify-between text-sm"><span>Thuế</span><span>{tax.toLocaleString('vi-VN')} đ</span></div>
                <div className="flex justify-between text-sm"><span>Giảm giá</span><span>{discount.toLocaleString('vi-VN')} đ</span></div>
                <div className="flex justify-between text-sm font-semibold"><span>Tổng</span><span>{total.toLocaleString('vi-VN')} đ</span></div>
              </div>
            </div>
          )}

          <div>
            <label className="text-sm font-medium">Phí khám (VND)</label>
            <input type="number" value={consultFee} onChange={e => setConsultFee(e.target.value)} className="w-full px-3 py-2 border rounded mt-1" />
          </div>

          <div className="flex justify-end gap-3 pt-4 border-t">
            <button onClick={onClose} className="px-4 py-2 border rounded">Hủy</button>
            <button onClick={handleCreateInvoice} disabled={loading} className="px-4 py-2 bg-green-600 text-white rounded">{loading ? 'Đang tạo...' : 'Tạo hóa đơn'}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
