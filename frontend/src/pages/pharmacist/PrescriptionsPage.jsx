/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import PharmacistLayout from '../../components/layout/PharmacistLayout';
import { prescriptionService } from '../../services/prescriptionService';
import PrescriptionDetailModal from '../../components/pharmacist/PrescriptionDetailModal';
import { Search, Edit2, Trash2 } from 'lucide-react';
import { PRESCRIPTION_STATUS_LABELS } from '../../utils/constants';

export default function PrescriptionsPage() {
  const [prescriptions, setPrescriptions] = useState([]);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(false);
  const location = useLocation();
  const [showDispenseModal, setShowDispenseModal] = useState(false);
  const [selectedPrescriptionId, setSelectedPrescriptionId] = useState(null);

  useEffect(() => {
    // read optional status query param (e.g. ?status=pending)
    const params = new URLSearchParams(location.search);
    const qStatus = params.get('status');
    if (qStatus) {
      setFilter(qStatus === 'pending' ? 'pending' : qStatus);
    }
    loadPrescriptions();
    // run when location.search changes (URL status param)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.search]);

  const loadPrescriptions = async (page = 1, limit = 50) => {
    setLoading(true);
    try {
      const res = await prescriptionService.getPrescriptions({ page, limit });
      console.log('getPrescriptions response:', res);
      
      let items = [];
      if (res && res.data && res.data.prescriptions && Array.isArray(res.data.prescriptions)) {
        items = res.data.prescriptions;
      } else if (res && res.prescriptions && Array.isArray(res.prescriptions)) {
        items = res.prescriptions;
      } else if (Array.isArray(res)) {
        items = res;
      }
      
      console.log('items:', items);
      
      setPrescriptions(items.map(p => {
        let medicines = [];
        try {
          const items_data = typeof p.items === 'string' ? JSON.parse(p.items) : p.items;
          medicines = Array.isArray(items_data) ? items_data : [];
        } catch (e) {
          medicines = [];
        }
        return {
          id: p.id,
          orderNo: `RX${String(p.id).padStart(4, '0')}`,
          patientName: p.patient?.user?.full_name || p.patient?.full_name || '',
          doctor: p.doctor?.user?.full_name || '',
          medicines: medicines,
          status: p.status,
          createdAt: p.created_at,
          raw: p,
        };
      }));
    } catch (err) {
      console.error('Failed to load prescriptions', err);
      setPrescriptions([]);
    } finally {
      setLoading(false);
    }
  };

  const filteredData = prescriptions.filter(p => {
    const matchesSearch = p.patientName.toLowerCase().includes(search.toLowerCase()) ||
                         p.orderNo.toLowerCase().includes(search.toLowerCase());

    let matchesFilter = true;
    if (filter === 'all') matchesFilter = true;
    else if (filter === 'pending') {
      // pending: APPROVED, INVOICED (ready for dispensing after payment), or READY_FOR_DISPENSE
      matchesFilter = (p.status === 'APPROVED' || p.status === 'INVOICED' || p.status === 'READY_FOR_DISPENSE');
    } else if (filter === 'dispensed') {
      matchesFilter = p.status === 'DISPENSED';
    } else if (filter === 'cancelled') {
      matchesFilter = p.status === 'CANCELLED';
    } else {
      // allow exact status match
      matchesFilter = p.status === filter;
    }

    return matchesSearch && matchesFilter;
  });

  const getStatusBadge = (status) => {
    const label = PRESCRIPTION_STATUS_LABELS[status] || status;
    const colorClass = status === 'DISPENSED' ? 'bg-green-100 text-green-800' : (status === 'APPROVED' || status === 'READY_FOR_DISPENSE' ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-800');
    return (
      <span className={`px-3 py-1 rounded-full text-sm font-medium ${colorClass}`}>
        {label}
      </span>
    );
  };

  return (
    <PharmacistLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Quản lý đơn thuốc</h1>
          <p className="text-gray-600 mt-2">Xử lý và theo dõi đơn thuốc từ bác sĩ</p>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search size={20} className="absolute left-3 top-3 text-gray-400" />
              <input
                type="text"
                placeholder="Tìm theo tên bệnh nhân hoặc số đơn..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
              />
            </div>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            >
              <option value="all">Tất cả</option>
              <option value="pending">Chưa xử lý</option>
              <option value="dispensed">Đã cấp</option>
              <option value="cancelled">Đã hủy</option>
            </select>
          </div>
        </div>

        {/* Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Số đơn</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Bệnh nhân</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Bác sĩ</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Thuốc</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Trạng thái</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Ngày</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase">Hành động</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {filteredData.map((p) => (
                  <tr key={p.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 text-sm font-medium text-blue-600">{p.orderNo}</td>
                    <td className="px-6 py-4 text-sm text-gray-900">{p.patientName}</td>
                    <td className="px-6 py-4 text-sm text-gray-600">{p.doctor}</td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      <div className="space-y-1 text-xs">
                        {Array.isArray(p.medicines) && p.medicines.length > 0 ? (
                          p.medicines.map((m, i) => (
                            <div key={i}>{m.medicine_name || m.name || 'N/A'} - {m.quantity || m.qty || 0}</div>
                          ))
                        ) : (
                          <span className="text-gray-400">Không có thuốc</span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">{getStatusBadge(p.status)}</td>
                    <td className="px-6 py-4 text-sm text-gray-600">{p.createdAt}</td>
                    <td className="px-6 py-4 text-sm">
                      <div className="flex gap-2">
                        <button className="text-blue-600 hover:text-blue-800 p-2">
                          <Edit2 size={16} />
                        </button>
                        {(p.status === 'APPROVED' || p.status === 'INVOICED' || p.status === 'READY_FOR_DISPENSE') && (
                          <button onClick={() => {
                            setSelectedPrescriptionId(p.id);
                            setShowDispenseModal(true);
                          }} className="text-green-600 hover:text-green-800 p-2" title="Bốc thuốc">
                            <Edit2 size={16} />
                          </button>
                        )}
                        <button className="text-red-600 hover:text-red-800 p-2">
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      {showDispenseModal && (
        <PrescriptionDetailModal
          prescriptionId={selectedPrescriptionId}
          onClose={() => { setShowDispenseModal(false); setSelectedPrescriptionId(null); loadPrescriptions(); }}
          onDispensed={(id) => { /* reload handled in onClose */ }}
        />
      )}
    </PharmacistLayout>
  );
}
