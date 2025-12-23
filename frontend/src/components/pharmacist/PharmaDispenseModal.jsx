/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
import React, { useEffect, useState, useMemo } from 'react';
import { X, Printer } from 'lucide-react';
import { prescriptionService } from '../../services/prescriptionService';
import { medicineService } from '../../services/medicineService';

export default function PharmaDispenseModal({ prescriptionId, onClose, onDispensed }) {
  const [loading, setLoading] = useState(false);
  const [prescription, setPrescription] = useState(null);
  const [items, setItems] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!prescriptionId) return;
    loadPrescription(prescriptionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prescriptionId]);

  const loadPrescription = async (id) => {
    try {
      setLoading(true);
      setError('');
      const res = await prescriptionService.getPrescriptionById(id);
      let pres = res?.data || res;
      if (pres && pres.data) pres = pres.data;
      if (pres && pres.prescription) pres = pres.prescription;
      setPrescription(pres);

      const rawItems = typeof pres.items === 'string' ? JSON.parse(pres.items) : pres.items || [];
      // enrich items with stock info
      const enriched = await Promise.all(rawItems.map(async (it) => {
        const medId = it.medicine_id || it.id || null;
        let stockQty = null;
        if (medId) {
          try {
            const res = await medicineService.getStockByMedicineId(medId);
            const data = res?.data || res;
            // endpoint returns { total_stock: number } or { data: { total_stock } }
            if (data?.data?.total_stock !== undefined) {
              stockQty = data.data.total_stock;
            } else if (data?.total_stock !== undefined) {
              stockQty = data.total_stock;
            } else {
              stockQty = 0;
            }
          } catch (e) {
            console.error('Failed to get stock for medicine', medId, e);
            stockQty = 0;
          }
        }

        return {
          medicine_id: medId,
          name: it.medicine_name || it.name || it.description || 'Thuốc',
          prescribed_qty: Number(it.quantity || it.qty || 1),
          dispense_qty: Number(it.quantity || it.qty || 1),
          unit_price: Number(it.unit_price || 0),
          stock_qty: stockQty,
        };
      }));
      setItems(enriched);
    } catch (err) {
      console.error('Failed to load prescription', err);
      setError(err?.response?.data?.message || err?.message || 'Không tải được đơn');
    } finally {
      setLoading(false);
    }
  };

  const updateQty = (index, value) => {
    setItems(prev => {
      const copy = [...prev];
      let v = parseInt(value, 10);
      if (isNaN(v) || v < 0) v = 0;
      // cap by prescribed qty
      if (v > copy[index].prescribed_qty) v = copy[index].prescribed_qty;
      copy[index] = { ...copy[index], dispense_qty: v };
      return copy;
    });
  };

  const handleConfirm = async () => {
    if (!prescription) return;
    if (!window.confirm('Xác nhận cấp thuốc và trừ kho theo đơn này?')) return;
    try {
      setLoading(true);
      // Build payload of dispense quantities for backend
      const payload = {
        items: items.map(it => ({ medicine_id: it.medicine_id, dispense_quantity: it.dispense_qty }))
      };
      await prescriptionService.dispensePrescription(prescription.id, payload);
      onDispensed && onDispensed(prescription.id);
      alert('Đã cấp thuốc.');
      onClose();
    } catch (err) {
      console.error('dispense error', err);
      alert('Lỗi khi cấp thuốc: ' + (err?.response?.data?.message || err?.message || ''));
    } finally {
      setLoading(false);
    }
  };

  const handlePrint = () => {
    // simple print: open a new window with a basic prescription summary
    const html = `
      <html>
        <head><title>In đơn thuốc</title></head>
        <body>
          <h2>Đơn thuốc #${prescription?.id}</h2>
          <p>Bệnh nhân: ${prescription?.patient?.user?.full_name || ''}</p>
          <p>Bác sĩ: ${prescription?.doctor?.user?.full_name || ''}</p>
          <table border="1" cellpadding="6" cellspacing="0">
            <tr><th>Thuốc</th><th>SL kê</th><th>SL cấp</th></tr>
            ${items.map(it => `<tr><td>${it.name}</td><td>${it.prescribed_qty}</td><td>${it.dispense_qty}</td></tr>`).join('')}
          </table>
        </body>
      </html>
    `;
    const w = window.open('', '_blank');
    if (w) {
      w.document.write(html);
      w.document.close();
      w.print();
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl">
        <div className="p-4 border-b flex items-center justify-between">
          <h3 className="text-lg font-semibold">Cấp thuốc cho đơn #{prescription?.id || prescriptionId}</h3>
          <div className="flex items-center gap-2">
            <button onClick={handlePrint} className="px-3 py-1 border rounded text-gray-600 hover:bg-gray-50"><Printer size={16} /> <span className="ml-2">In</span></button>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-800"><X /></button>
          </div>
        </div>
        <div className="p-4 space-y-4">
          {error && <div className="p-3 bg-red-50 text-red-800 rounded">{error}</div>}

          {loading && <div className="text-sm text-gray-500">Đang tải...</div>}

          {prescription && (
            <div>
              <div className="flex justify-between text-sm mb-2"><span>Bệnh nhân</span><span>{prescription.patient?.user?.full_name || prescription.patient?.full_name}</span></div>
              <div className="flex justify-between text-sm mb-2"><span>Bác sĩ</span><span>{prescription.doctor?.user?.full_name}</span></div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left">
                      <th className="py-1">Thuốc</th>
                      <th className="py-1">SL kê</th>
                      <th className="py-1">SL cấp</th>
                      <th className="py-1">Tồn kho</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((it, i) => (
                      <tr key={i} className="align-top">
                        <td className="py-1">{it.name}</td>
                        <td className="py-1">{it.prescribed_qty}</td>
                        <td className="py-1"><input type="number" value={it.dispense_qty} min={0} max={it.prescribed_qty} onChange={e => updateQty(i, e.target.value)} className="w-20 px-2 py-1 border rounded" /></td>
                        <td className="py-1">
                          {it.stock_qty !== null && it.stock_qty !== undefined ? it.stock_qty : '–'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>


            </div>
          )}

          <div className="flex justify-end gap-3 pt-4 border-t">
            <button onClick={onClose} className="px-4 py-2 border rounded">Hủy</button>
            <button onClick={handleConfirm} disabled={loading} className="px-4 py-2 bg-green-600 text-white rounded">{loading ? 'Đang xử lý...' : 'Xác nhận cấp thuốc'}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
