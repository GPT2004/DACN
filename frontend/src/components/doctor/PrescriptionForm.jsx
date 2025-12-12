// src/components/doctor/PrescriptionForm.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Pill, Plus, X } from 'lucide-react';
import { medicineService } from '../../services/medicineService';

const PrescriptionForm = ({ patient, onSave, onCancel, initialPrescriptions }) => {
  const [medicines, setMedicines] = useState([
    { id: Date.now(), medicine_id: null, name: '', unit_price: 0, dosage: '', quantity: '', duration: '', instructions: '' }
  ]);
  const [notes, setNotes] = useState('');
  const [allMedicines, setAllMedicines] = useState([]);
  const [filteredByRow, setFilteredByRow] = useState({});
  const [showExistingPrescriptions, setShowExistingPrescriptions] = useState(false);
  const debounceTimers = useRef({});
  const inputRefs = useRef({});

  const addMedicine = () => {
    setMedicines([...medicines, {
      id: Date.now(),
      medicine_id: null,
      name: '', unit_price: 0, dosage: '', quantity: '', duration: '', instructions: ''
    }]);
  };

  const removeMedicine = (id) => {
    setMedicines(prev => prev.filter(m => m.id !== id));
    // keep refs tidy
    if (inputRefs.current[id]) delete inputRefs.current[id];
  };

  const updateMedicine = (id, field, value) => {
    setMedicines(prev => prev.map(m => m.id === id ? { ...m, [field]: value } : m));
  };

  useEffect(() => {
    let mounted = true;
    // load medicines list for typeahead (limit to 500 to avoid huge payloads)
    medicineService.getMedicines({ limit: 500 }).then(res => {
      const data = res?.data?.data || res?.data || res || [];
      if (!mounted) return;
      // normalize to array of { id, name, unit_price }
      const list = Array.isArray(data) ? data : (data?.medicines || []);
      setAllMedicines(list.map(m => ({ id: m.id, name: m.name || m.medicine_name || m.title || '', unit_price: m.unit_price || 0 })));
    }).catch(() => {});
    return () => { mounted = false; };
  }, []);

  // Show existing prescriptions by default if they exist
  useEffect(() => {
    if (initialPrescriptions && initialPrescriptions.length > 0) {
      setShowExistingPrescriptions(true);
    }
  }, [initialPrescriptions]);

  // close dropdowns when clicking outside any prescription row
  useEffect(() => {
    const onDoc = (e) => {
      // if click is inside any element with data-prescription-row, do nothing
      const inside = e.target.closest && e.target.closest('[data-prescription-row]');
      if (!inside) setFilteredByRow({});
    };
    document.addEventListener('click', onDoc);
    return () => document.removeEventListener('click', onDoc);
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Ensure every medicine has a medicine_id. If user typed a name but didn't select,
    // try to match it against the loaded `allMedicines` list (case-insensitive).
    const processed = medicines.map(m => {
      let medicine_id = m.medicine_id;
      if (!medicine_id && m.name) {
        const found = allMedicines.find(a => (a.name || '').toLowerCase() === (m.name || '').toLowerCase());
        if (found) medicine_id = found.id;
      }
      return { ...m, medicine_id };
    });

    const missing = processed.find(p => !p.medicine_id);
    if (missing) {
      // Focus user on the first missing row and show a helpful message
      alert('Vui lòng chọn thuốc từ danh sách gợi ý cho mỗi hàng (click để chọn).');
      return;
    }

    onSave({ medicines: processed, notes });
  };

  return (
    <div className="bg-white rounded-lg shadow-lg">
      <div className="p-6 border-b">
        <h3 className="text-xl font-bold text-gray-900">Kê Đơn Thuốc</h3>
        <p className="text-sm text-gray-600 mt-1">Bệnh nhân: {patient?.name}</p>
      </div>

      {/* Existing Prescriptions Section */}
      {initialPrescriptions && initialPrescriptions.length > 0 && (
        <div className="p-6 border-b bg-blue-50">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-gray-900">📋 Đơn thuốc cũ ({initialPrescriptions.length})</h4>
            <button
              type="button"
              onClick={() => setShowExistingPrescriptions(!showExistingPrescriptions)}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              {showExistingPrescriptions ? '▼ Ẩn' : '▶ Hiện'}
            </button>
          </div>
          
          {showExistingPrescriptions && (
            <div className="space-y-3 mt-3">
              {initialPrescriptions.map((p, pIdx) => (
                <div key={p.id} className="bg-white rounded p-3 border border-blue-200">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <div className="text-sm font-medium text-gray-900">Đơn #{p.id}</div>
                      <div className="text-xs text-gray-500">
                        {p.created_at ? new Date(p.created_at).toLocaleDateString('vi-VN') : ''}
                      </div>
                    </div>
                    <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                      {p.status || 'DRAFT'}
                    </span>
                  </div>
                  
                  {p.items && Array.isArray(p.items) && p.items.length > 0 && (
                    <div className="space-y-2 mb-2">
                      {p.items.map((item, iIdx) => (
                        <div key={iIdx} className="text-sm">
                          <div className="font-medium text-gray-800">{item.medicine_name}</div>
                          <div className="text-xs text-gray-600">
                            Liều: {item.dosage} | SL: {item.quantity} {item.instructions && `| HD: ${item.instructions}`}
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              setMedicines([...medicines, {
                                id: Date.now(),
                                medicine_id: item.medicine_id,
                                name: item.medicine_name,
                                unit_price: item.unit_price || 0,
                                dosage: item.dosage,
                                quantity: item.quantity,
                                duration: '',
                                instructions: item.instructions
                              }]);
                            }}
                            className="text-xs text-emerald-600 hover:text-emerald-800 mt-1"
                          >
                            ➕ Dùng lại
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <form onSubmit={handleSubmit} className="p-6 space-y-6">
        <div className="space-y-4">
          {medicines.map((medicine, index) => (
            <div key={medicine.id} data-prescription-row={medicine.id} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
              <div className="flex justify-between items-center mb-3">
                <h4 className="font-medium text-gray-900">Thuốc {index + 1}</h4>
                {medicines.length > 1 && (
                  <button type="button" onClick={() => removeMedicine(medicine.id)} className="text-red-600 hover:text-red-700">
                    <X className="w-5 h-5" />
                  </button>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Tên thuốc *</label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                    placeholder="VD: Amoxicillin 500mg"
                    value={medicine.name}
                    ref={el => { inputRefs.current[medicine.id] = el; }}
                    onChange={(e) => {
                      const v = e.target.value;
                      updateMedicine(medicine.id, 'name', v);
                      // no input -> clear
                      if (!v) {
                        setFilteredByRow(prev => ({ ...prev, [medicine.id]: [] }));
                        return;
                      }

                      const q = v.toLowerCase();

                      // If we have a local list, filter client-side
                      if (allMedicines && allMedicines.length > 0) {
                        const matches = allMedicines.filter(m => m.name.toLowerCase().includes(q)).slice(0, 8);
                        setFilteredByRow(prev => ({ ...prev, [medicine.id]: matches }));
                        return;
                      }

                      // Otherwise, debounce a server-side search
                      if (debounceTimers.current[medicine.id]) clearTimeout(debounceTimers.current[medicine.id]);
                      debounceTimers.current[medicine.id] = setTimeout(() => {
                        medicineService.getMedicines({ q: v, limit: 10 }).then(res => {
                          const data = res?.data?.data || res?.data || res || [];
                          const list = Array.isArray(data) ? data : (data?.medicines || []);
                          const matches = list.map(m => ({ id: m.id, name: m.name || m.medicine_name || '', unit_price: m.unit_price || 0 }));
                          setFilteredByRow(prev => ({ ...prev, [medicine.id]: matches }));
                        }).catch(() => {
                          setFilteredByRow(prev => ({ ...prev, [medicine.id]: [] }));
                        });
                      }, 250);
                    }}
                    required
                  />

                  {filteredByRow[medicine.id] && filteredByRow[medicine.id].length > 0 && (
                    <div className="mt-1 border bg-white rounded shadow max-h-56 overflow-auto z-50">
                      {filteredByRow[medicine.id].map(opt => (
                        <div
                          key={opt.id}
                          className="px-3 py-2 hover:bg-gray-100 cursor-pointer"
                          onMouseDown={(e) => {
                            // prevent input blur before we set the value
                            e.preventDefault();
                            updateMedicine(medicine.id, 'name', opt.name);
                            updateMedicine(medicine.id, 'medicine_id', opt.id);
                            updateMedicine(medicine.id, 'unit_price', opt.unit_price || 0);
                            setFilteredByRow(prev => ({ ...prev, [medicine.id]: [] }));
                            // ensure the input reflects the updated value and keep focus
                            setTimeout(() => {
                              const el = inputRefs.current[medicine.id];
                              if (el) {
                                try { el.focus(); el.setSelectionRange(el.value.length, el.value.length); } catch (err) {}
                              }
                            }, 0);
                          }}
                        >
                          <div className="text-sm font-medium">{opt.name}</div>
                          <div className="text-xs text-gray-500">{opt.unit_price ? (opt.unit_price.toLocaleString('vi-VN') + ' VND') : ''}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Liều lượng *</label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                    placeholder="VD: 1 viên/lần"
                    value={medicine.dosage}
                    onChange={(e) => updateMedicine(medicine.id, 'dosage', e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Số lượng *</label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                    placeholder="VD: 2 hộp (20 viên)"
                    value={medicine.quantity}
                    onChange={(e) => updateMedicine(medicine.id, 'quantity', e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Thời gian dùng</label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                    placeholder="VD: 7 ngày"
                    value={medicine.duration}
                    onChange={(e) => updateMedicine(medicine.id, 'duration', e.target.value)}
                  />
                </div>
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Hướng dẫn sử dụng</label>
                  <textarea
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                    rows="2"
                    placeholder="VD: Uống sau bữa ăn, ngày 3 lần"
                    value={medicine.instructions}
                    onChange={(e) => updateMedicine(medicine.id, 'instructions', e.target.value)}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
        <button
          type="button"
          onClick={addMedicine}
          className="w-full px-4 py-2 border-2 border-dashed border-gray-300 text-gray-600 rounded-lg hover:border-blue-500 hover:text-blue-600 flex items-center justify-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Thêm thuốc
        </button>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Ghi chú</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            rows="3"
            placeholder="Lời dặn thêm..."
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </div>
        <div className="flex justify-end gap-3 pt-4 border-t">
          <button type="button" onClick={onCancel} className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50">
            Hủy
          </button>
          <button type="submit" className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2">
            <Pill className="w-4 h-4" />
            Lưu đơn thuốc
          </button>
        </div>
      </form>
    </div>
  );
};

export default PrescriptionForm;