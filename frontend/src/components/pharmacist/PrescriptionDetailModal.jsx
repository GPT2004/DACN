/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react';
import { X, Eye, Download } from 'lucide-react';
import { prescriptionService } from '../../services/prescriptionService';
import { invoiceService } from '../../services/invoiceService';
import PharmaDispenseModal from './PharmaDispenseModal';

export default function PrescriptionDetailModal({ prescriptionId, onClose, onDispensed }) {
  const [prescription, setPrescription] = useState(null);
  const [invoice, setInvoice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showDispenseModal, setShowDispenseModal] = useState(false);
  const [printing, setPrinting] = useState(false);

  useEffect(() => {
    if (!prescriptionId) return;
    loadDetails(prescriptionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prescriptionId]);

  const loadDetails = async (id) => {
    setLoading(true);
    try {
      // Load prescription
      const presRes = await prescriptionService.getPrescriptionById(id);
      let prescData = presRes?.data || presRes;
      if (prescData && prescData.data) prescData = prescData.data;
      if (prescData && prescData.prescription) prescData = prescData.prescription;
      setPrescription(prescData);

      // Try to load related invoice (if any)
      if (prescData?.id) {
        try {
          const invRes = await invoiceService.getInvoices({ prescription_id: id, page: 1, limit: 1 });
          let invItems = [];
          if (invRes && invRes.data && invRes.data.invoices && Array.isArray(invRes.data.invoices)) {
            invItems = invRes.data.invoices;
          } else if (invRes && invRes.invoices && Array.isArray(invRes.invoices)) {
            invItems = invRes.invoices;
          }
          if (invItems.length > 0) {
            setInvoice(invItems[0]);
          }
        } catch (e) {
          // no invoice found, that's ok
        }
      }
    } catch (err) {
      console.error('Failed to load prescription details', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDispense = () => {
    setShowDispenseModal(true);
  };

  const handlePrint = () => {
    setPrinting(true);
    window.print();
    setTimeout(() => setPrinting(false), 1000);
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6">Đang tải...</div>
      </div>
    );
  }

  if (!prescription) {
    return null;
  }

  return (
    <>
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
          <div className="sticky top-0 bg-gradient-to-r from-blue-600 to-blue-700 text-white p-6 flex justify-between items-center">
            <h2 className="text-xl font-bold">Chi tiết đơn thuốc & hóa đơn</h2>
            <button
              onClick={() => { 
                setShowDispenseModal(false);
                onClose && onClose();
              }}
              className="text-white hover:text-gray-200 text-2xl"
            >
              ×
            </button>
          </div>

          <div className="p-6 space-y-6">
            {/* Invoice Info (if exists) */}
            {invoice && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Thông tin hóa đơn</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Mã hóa đơn</p>
                    <p className="text-lg font-medium">HĐ{String(invoice.id).padStart(6, '0')}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Trạng thái</p>
                    <p className="text-lg font-medium text-green-600">Đã thanh toán</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Bệnh nhân</p>
                    <p className="text-lg font-medium">{invoice.patient?.user?.full_name || invoice.patient?.full_name || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">SĐT</p>
                    <p className="text-lg font-medium">{invoice.patient?.user?.phone || invoice.patient?.phone || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Tổng tiền</p>
                    <p className="text-lg font-medium text-green-600">{invoice.total?.toLocaleString() || 0} VND</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Ngày thanh toán</p>
                    <p className="text-lg font-medium">{invoice.paid_at ? new Date(invoice.paid_at).toLocaleDateString('vi-VN') : 'N/A'}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Invoice Items */}
            {invoice && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Danh sách dịch vụ/thuốc</h3>
                <div className="border border-gray-200 rounded-lg overflow-hidden">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-600">Mô tả</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-600">SL</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-600">Đơn giá</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-600">Thành tiền</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {/* Show consultation fee if exists */}
                      {invoice.items && invoice.items.length > 0 && invoice.items[0].type === 'consultation' && (
                        <tr>
                          <td className="px-4 py-2">{invoice.items[0].description}</td>
                          <td className="px-4 py-2">{invoice.items[0].quantity}</td>
                          <td className="px-4 py-2">{invoice.items[0].unit_price?.toLocaleString() || 0} VND</td>
                          <td className="px-4 py-2 font-medium">{invoice.items[0].amount?.toLocaleString() || 0} VND</td>
                        </tr>
                      )}
                      
                      {/* Show prescription items (medicines) */}
                      {prescription && (typeof prescription.items === 'string' 
                        ? JSON.parse(prescription.items) 
                        : prescription.items || []
                      ).map((item, i) => {
                        const quantity = Number(item.quantity) || 0;
                        const unitPrice = Number(item.unit_price) || 0;
                        const amount = quantity * unitPrice;
                        
                        // If medicine_name is Unknown, show as "Phí thuốc"
                        const displayName = (item.medicine_name === 'Unknown' || !item.medicine_name) 
                          ? 'Phí thuốc' 
                          : item.medicine_name;
                        
                        return (
                          <tr key={i}>
                            <td className="px-4 py-2">{displayName}</td>
                            <td className="px-4 py-2">{quantity}</td>
                            <td className="px-4 py-2">{unitPrice.toLocaleString()} VND</td>
                            <td className="px-4 py-2 font-medium">{amount.toLocaleString()} VND</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Prescription Info */}
            {prescription && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Đơn thuốc (ĐT#{prescription.id})</h3>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-3">
                  <div>
                    <p className="text-sm text-gray-600">Bác sĩ</p>
                    <p className="font-medium text-lg">{prescription.doctor?.user?.full_name || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600 mb-3">Danh sách thuốc cần cấp</p>
                    <div className="space-y-2 border-t border-blue-200 pt-3">
                      {(typeof prescription.items === 'string'
                        ? JSON.parse(prescription.items)
                        : prescription.items || []
                      ).map((item, i) => (
                        <div key={i} className="bg-white p-3 rounded border border-gray-200">
                          <div className="font-semibold text-gray-900">{item.medicine_name || 'N/A'}</div>
                          <div className="text-sm text-gray-600 mt-1">Số lượng: <span className="font-medium text-gray-900">{item.quantity || 0}</span></div>
                          {item.dosage && <div className="text-sm text-gray-600">Liều dùng: {item.dosage}</div>}
                          {item.instructions && <div className="text-sm text-gray-600">Hướng dẫn: {item.instructions}</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3 border-t border-gray-200 pt-6">
              <button
                onClick={() => {
                  setShowDispenseModal(false);
                  onClose && onClose();
                }}
                className="flex-1 px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 font-medium"
              >
                Đóng
              </button>
              <button
                onClick={handlePrint}
                disabled={printing}
                className="flex items-center justify-center gap-2 flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 font-medium"
              >
                <Download className="w-4 h-4" />
                {printing ? 'Đang in...' : 'In đơn'}
              </button>
              {prescription && prescription.status !== 'DISPENSED' && (
                <button
                  onClick={handleDispense}
                  className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
                >
                  Bốc thuốc
                </button>
              )}
              {prescription && prescription.status === 'DISPENSED' && (
                <div className="flex-1 px-4 py-2 bg-gray-100 text-gray-600 rounded-lg font-medium text-center">
                  ✓ Đã cấp thuốc
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Dispense Modal */}
      {showDispenseModal && prescription && (
        <PharmaDispenseModal
          prescriptionId={prescription.id}
          onClose={() => {
            setShowDispenseModal(false);
            onClose && onClose();
          }}
          onDispensed={() => {
            setShowDispenseModal(false);
            onDispensed && onDispensed(prescription.id);
            onClose && onClose();
          }}
        />
      )}
    </>
  );
}
