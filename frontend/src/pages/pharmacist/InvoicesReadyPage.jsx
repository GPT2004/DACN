import React, { useState, useEffect } from 'react';
import PharmacistLayout from '../../components/layout/PharmacistLayout';
import { invoiceService } from '../../services/invoiceService';
import { prescriptionService } from '../../services/prescriptionService';
import PrescriptionDetailModal from '../../components/pharmacist/PrescriptionDetailModal';
import { Search } from 'lucide-react';

export default function InvoicesReadyPage() {
  const [invoices, setInvoices] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedPrescriptionId, setSelectedPrescriptionId] = useState(null);

  useEffect(() => {
    loadInvoices();
  }, []);

  const loadInvoices = async () => {
    setLoading(true);
    try {
      const res = await invoiceService.getInvoices({ status: 'PAID', page: 1, limit: 50 });
      
      let items = [];
      if (res && res.data && res.data.invoices && Array.isArray(res.data.invoices)) {
        items = res.data.invoices;
      } else if (res && res.invoices && Array.isArray(res.invoices)) {
        items = res.invoices;
      } else if (Array.isArray(res)) {
        items = res;
      }
      
      // Enrich with prescription status
      const enriched = await Promise.all(items.map(async (inv) => {
        let prescStatus = null;
        if (inv.prescription_id) {
          try {
            const presRes = await prescriptionService.getPrescriptionById(inv.prescription_id);
            let presData = presRes?.data || presRes;
            if (presData && presData.data) presData = presData.data;
            if (presData && presData.prescription) presData = presData.prescription;
            prescStatus = presData?.status;
          } catch (e) {
            // ignore
          }
        }
        return {
          ...inv,
          prescription_status: prescStatus
        };
      }));
      
      setInvoices(enriched);
    } catch (err) {
      setInvoices([]);
    } finally {
      setLoading(false);
    }
  };

  const handleViewPrescription = async (prescriptionId) => {
    if (!prescriptionId) {
      alert('Không tìm thấy đơn thuốc cho hóa đơn này');
      return;
    }
    setSelectedPrescriptionId(prescriptionId);
  };

  const filteredData = invoices.filter(inv => {
    const patientName = inv.patient?.user?.full_name || inv.patient?.full_name || '';
    const invoiceId = `HĐ${String(inv.id).padStart(6, '0')}`;
    return patientName.toLowerCase().includes(search.toLowerCase()) ||
           invoiceId.toLowerCase().includes(search.toLowerCase());
  });

  return (
    <PharmacistLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Đơn thuốc</h1>
          <p className="text-gray-600 mt-2">Danh sách đơn thuốc sẵn sàng để bốc</p>
        </div>

        {/* Search */}
        <div className="bg-white rounded-lg shadow p-4">
          <div className="relative">
            <Search size={20} className="absolute left-3 top-3 text-gray-400" />
            <input
              type="text"
              placeholder="Tìm theo tên bệnh nhân hoặc mã hóa đơn..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>

        {/* Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          {loading ? (
            <div className="p-6 text-center text-gray-500">Đang tải...</div>
          ) : filteredData.length === 0 ? (
            <div className="p-6 text-center text-gray-500">Không có hóa đơn nào sẵn sàng</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Mã HĐ</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Bệnh nhân</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">SĐT</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Tổng tiền</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Ngày thanh toán</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Trạng thái</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Hành động</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {filteredData.map((inv) => (
                    <tr key={inv.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 text-sm font-medium text-blue-600">HĐ{String(inv.id).padStart(6, '0')}</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{inv.patient?.user?.full_name || inv.patient?.full_name || 'N/A'}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{inv.patient?.user?.phone || inv.patient?.phone || 'N/A'}</td>
                      <td className="px-6 py-4 text-sm font-medium text-green-600">{inv.total?.toLocaleString() || 0} VND</td>
                      <td className="px-6 py-4 text-sm text-gray-600">
                        {inv.paid_at ? new Date(inv.paid_at).toLocaleDateString('vi-VN') : 'N/A'}
                      </td>
                      <td className="px-6 py-4 text-sm">
                        {inv.prescription_status === 'DISPENSED' ? (
                          <span className="px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                            ✓ Đã bốc
                          </span>
                        ) : (
                          <span className="px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800">
                            Chưa bốc
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-sm">
                        <button
                          onClick={() => handleViewPrescription(inv.prescription_id)}
                          disabled={!inv.prescription_id}
                          className={`font-medium ${
                            inv.prescription_id
                              ? 'text-blue-600 hover:text-blue-800 cursor-pointer'
                              : 'text-gray-400 cursor-not-allowed'
                          }`}
                          title={inv.prescription_id ? 'Xem chi tiết để cấp' : 'Không có đơn thuốc'}
                        >
                          Xem đơn
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Prescription Detail Modal */}
      {selectedPrescriptionId && (
        <PrescriptionDetailModal
          prescriptionId={selectedPrescriptionId}
          onClose={() => { 
            setSelectedPrescriptionId(null);
            loadInvoices();
          }}
          onDispensed={() => { 
            setSelectedPrescriptionId(null);
            loadInvoices();
          }}
        />
      )}
    </PharmacistLayout>
  );
}
