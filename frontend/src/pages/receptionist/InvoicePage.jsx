import React, { useEffect, useState } from 'react';
import { DollarSign, Plus, Search, Filter, Eye, Printer, CheckCircle, X } from 'lucide-react';
import { prescriptionService } from '../../services/prescriptionService';
import PaymentForm from '../../components/receptionist/PaymentForm';
import { invoiceService } from '../../services/invoiceService';
import CreateInvoiceFromPrescriptionModal from '../../components/receptionist/CreateInvoiceFromPrescriptionModal';

export default function InvoicePage() {
  const [invoices, setInvoices] = useState([]);
  const [filteredInvoices, setFilteredInvoices] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [showPaymentForm, setShowPaymentForm] = useState(false);
  const [selectedInvoice, setSelectedInvoice] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [pendingPrescriptions, setPendingPrescriptions] = useState([]);
  const [showCreateFromPrescriptionModal, setShowCreateFromPrescriptionModal] = useState(false);
  const [modalPrescriptionId, setModalPrescriptionId] = useState(null);

  // Mock data - thay thế bằng API thực tế
  const mockInvoices = [
    {
      id: 1,
      invoice_no: 'INV-2025-001',
      patient: { user: { full_name: 'Nguyễn Văn A', phone: '0901234567' } },
      appointment: { doctor: { user: { full_name: 'TS. Trần B' } } },
      subtotal: 500000,
      tax: 50000,
      discount: 0,
      total: 550000,
      status: 'UNPAID',
      items: [
        { id: 1, description: 'Khám tổng quát', quantity: 1, unit_price: 200000, total: 200000 },
        { id: 2, description: 'Xét nghiệm máu', quantity: 1, unit_price: 300000, total: 300000 }
      ],
      created_at: new Date().toISOString(),
      paid_at: null
    },
    {
      id: 2,
      invoice_no: 'INV-2025-002',
      patient: { user: { full_name: 'Phạm Thị C', phone: '0912345678' } },
      appointment: { doctor: { user: { full_name: 'TS. Nguyễn D' } } },
      subtotal: 300000,
      tax: 30000,
      discount: 0,
      total: 330000,
      status: 'PAID',
      items: [
        { id: 1, description: 'Khám ngoài', quantity: 1, unit_price: 300000, total: 300000 }
      ],
      created_at: new Date(Date.now() - 86400000).toISOString(),
      paid_at: new Date(Date.now() - 86400000).toISOString()
    },
    {
      id: 3,
      invoice_no: 'INV-2025-003',
      patient: { user: { full_name: 'Trần Văn E', phone: '0923456789' } },
      appointment: { doctor: { user: { full_name: 'TS. Lê F' } } },
      subtotal: 750000,
      tax: 75000,
      discount: 50000,
      total: 775000,
      status: 'UNPAID',
      items: [
        { id: 1, description: 'Khám nhi', quantity: 1, unit_price: 250000, total: 250000 },
        { id: 2, description: 'Siêu âm', quantity: 1, unit_price: 500000, total: 500000 }
      ],
      created_at: new Date(Date.now() - 172800000).toISOString(),
      paid_at: null
    }
  ];

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setInvoices(mockInvoices);
      setLoading(false);
    }, 500);
    // fetch prescriptions waiting for invoicing
    (async () => {
      try {
        const res = await prescriptionService.getForInvoicing({ page: 1, limit: 10 });
        const data = res?.data?.data || res?.data || res;
        setPendingPrescriptions(data.prescriptions || []);
      } catch (err) {
        console.error('Failed to fetch pending prescriptions', err);
      }
    })();
  }, []);

  // Filter invoices
  useEffect(() => {
    let filtered = invoices;

    if (searchTerm) {
      filtered = filtered.filter(inv =>
        inv.invoice_no.toLowerCase().includes(searchTerm.toLowerCase()) ||
        inv.patient?.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        inv.patient?.user?.phone?.includes(searchTerm)
      );
    }

    if (statusFilter) {
      filtered = filtered.filter(inv => inv.status === statusFilter);
    }

    setFilteredInvoices(filtered);
  }, [invoices, searchTerm, statusFilter]);

  const handlePayment = (invoiceId) => {
    const invoice = invoices.find(i => i.id === invoiceId);
    setSelectedInvoice(invoice);
    setShowPaymentForm(true);
  };

  // Normalize selectedInvoice (sometimes API returns wrapper { success, message, data: invoice })
  const normalizedSelectedInvoice = selectedInvoice ? (selectedInvoice.data || selectedInvoice.invoice || selectedInvoice) : null;

  const handlePaymentConfirm = async (paymentData) => {
    try {
      // call backend to pay invoice
      const paidAmount = paymentData.netPaid || paymentData.paidAmount || 0;
      await invoiceService.payInvoice(selectedInvoice.id, paymentData.method?.toLowerCase?.() || 'cash', paidAmount);

      // Update invoice status locally
      const updatedInvoices = invoices.map(inv =>
        inv.id === selectedInvoice.id
          ? { ...inv, status: 'PAID', paid_at: new Date().toISOString() }
          : inv
      );
      setInvoices(updatedInvoices);
      setShowPaymentForm(false);
      setSelectedInvoice(null);
      alert('Thanh toán thành công!');
    } catch (err) {
      console.error('Payment failed', err);
      alert('Thanh toán thất bại: ' + (err?.response?.data?.message || err.message || 'Unknown error'));
    }
  };

  const handlePrint = (invoice) => {
    // Print implementation
    const content = `
      PHIẾU THANH TOÁN
      ${invoice.invoice_no}
      
      Bệnh nhân: ${invoice.patient.user.full_name}
      Bác sĩ: ${invoice.appointment.doctor.user.full_name}
      
      Chi tiết:
      ${invoice.items.map(item => `${item.description}: ${item.total.toLocaleString('vi-VN')} VND`).join('\n')}
      
      Cộng: ${invoice.subtotal.toLocaleString('vi-VN')} VND
      Thuế: ${invoice.tax.toLocaleString('vi-VN')} VND
      Giảm giá: ${invoice.discount.toLocaleString('vi-VN')} VND
      Tổng: ${invoice.total.toLocaleString('vi-VN')} VND
      
      Trạng thái: ${invoice.status === 'PAID' ? 'Đã thanh toán' : 'Chưa thanh toán'}
    `;
    window.print();
  };

  const getStatusColor = (status) => {
    return status === 'PAID'
      ? 'bg-green-100 text-green-800'
      : 'bg-orange-100 text-orange-800';
  };

  const getStatusLabel = (status) => {
    return status === 'PAID' ? 'Đã thanh toán' : 'Chưa thanh toán';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
            <DollarSign className="w-8 h-8 text-green-600" />
            Quản Lý Hóa Đơn
          </h1>
          <p className="text-gray-600 mt-1">Quản lý và thanh toán hóa đơn bệnh nhân</p>
        </div>
      </div>

      {/* Search & Filter */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="relative">
            <Search className="w-5 h-5 absolute left-3 top-3 text-gray-400" />
            <input
              type="text"
              placeholder="Tìm theo số hóa đơn, tên bệnh nhân..."
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-gray-600" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            >
              <option value="">Tất cả trạng thái</option>
              <option value="UNPAID">Chưa thanh toán</option>
              <option value="PAID">Đã thanh toán</option>
            </select>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600 text-sm">Tổng hóa đơn</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">{invoices.length}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600 text-sm">Chưa thanh toán</p>
          <p className="text-3xl font-bold text-orange-600 mt-2">
            {invoices.filter(i => i.status === 'UNPAID').length}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600 text-sm">Tổng nợ</p>
          <p className="text-3xl font-bold text-red-600 mt-2">
            {invoices.filter(i => i.status === 'UNPAID').reduce((sum, i) => sum + i.total, 0).toLocaleString('vi-VN')} ₫
          </p>
        </div>
        {/* Pending prescriptions card */}
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600 text-sm">Đơn thuốc chờ lập hoá đơn</p>
          <p className="text-2xl font-bold text-gray-900 mt-2">{pendingPrescriptions.length}</p>
                <div className="mt-3">
            {pendingPrescriptions.slice(0,3).map(p => (
              <div key={p.id} className="flex items-center justify-between py-2">
                <div className="text-sm">Đơn #{p.id} — {p.patient?.user?.full_name || 'N/A'}</div>
                <button onClick={() => { setModalPrescriptionId(p.id); setShowCreateFromPrescriptionModal(true); }} className="text-sm px-2 py-1 bg-green-600 text-white rounded">Tạo</button>
              </div>
            ))}
            {pendingPrescriptions.length > 3 && (
              <div className="text-xs text-gray-500 mt-2">... và {pendingPrescriptions.length - 3} đơn khác</div>
            )}
          </div>
        </div>
      </div>

      {/* Payment Modal */}
      {showPaymentForm && normalizedSelectedInvoice && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900">Thanh Toán Hóa Đơn</h2>
              <button
                onClick={() => {
                  setShowPaymentForm(false);
                  setSelectedInvoice(null);
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            <div className="mb-4 p-3 bg-gray-50 rounded">
              <p className="text-sm text-gray-600">Hóa đơn: {normalizedSelectedInvoice.invoice_no || normalizedSelectedInvoice.id}</p>
              <p className="text-sm text-gray-600">Bệnh nhân: {normalizedSelectedInvoice.patient?.user?.full_name || normalizedSelectedInvoice.patient?.full_name}</p>
              <p className="text-lg font-bold text-gray-900 mt-2">
                {(Number(normalizedSelectedInvoice.total ?? normalizedSelectedInvoice.subtotal ?? normalizedSelectedInvoice.total_amount) || 0).toLocaleString('vi-VN')} ₫
              </p>
            </div>
            <PaymentForm
              amount={Number(normalizedSelectedInvoice.total ?? normalizedSelectedInvoice.subtotal ?? normalizedSelectedInvoice.total_amount) || 0}
              alreadyPaid={Number(normalizedSelectedInvoice.paid_amount ?? (normalizedSelectedInvoice.status === 'PAID' ? (normalizedSelectedInvoice.total ?? normalizedSelectedInvoice.subtotal ?? 0) : 0)) || 0}
              onPay={(paymentData) => handlePaymentConfirm(paymentData)}
            />
          </div>
        </div>
      )}

      {/* Detail Modal */}
      {showDetailModal && selectedInvoice && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-900">{selectedInvoice.invoice_no}</h2>
              <button
                onClick={() => {
                  setShowDetailModal(false);
                  setSelectedInvoice(null);
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Invoice Details */}
            <div className="space-y-4 mb-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Bệnh nhân</p>
                  <p className="text-lg font-semibold text-gray-900">{selectedInvoice.patient.user.full_name}</p>
                  <p className="text-sm text-gray-600">{selectedInvoice.patient.user.phone}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Bác sĩ</p>
                  <p className="text-lg font-semibold text-gray-900">{selectedInvoice.appointment.doctor.user.full_name}</p>
                </div>
              </div>

              {/* Items */}
              <div className="border-t pt-4">
                <h3 className="font-semibold text-gray-900 mb-3">Chi tiết dịch vụ</h3>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 text-gray-600">Dịch vụ</th>
                      <th className="text-right py-2 text-gray-600">Số lượng</th>
                      <th className="text-right py-2 text-gray-600">Đơn giá</th>
                      <th className="text-right py-2 text-gray-600">Thành tiền</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedInvoice.items.map((item) => (
                      <tr key={item.id} className="border-b">
                        <td className="py-2 text-gray-900">{item.description}</td>
                        <td className="py-2 text-right text-gray-600">{item.quantity}</td>
                        <td className="py-2 text-right text-gray-600">
                          {item.unit_price.toLocaleString('vi-VN')} ₫
                        </td>
                        <td className="py-2 text-right font-semibold text-gray-900">
                          {item.total.toLocaleString('vi-VN')} ₫
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Totals */}
              <div className="bg-gray-50 rounded p-4 space-y-2">
                <div className="flex justify-between text-gray-900">
                  <span>Cộng:</span>
                  <span className="font-semibold">{selectedInvoice.subtotal.toLocaleString('vi-VN')} ₫</span>
                </div>
                <div className="flex justify-between text-gray-900">
                  <span>Thuế (10%):</span>
                  <span className="font-semibold">{selectedInvoice.tax.toLocaleString('vi-VN')} ₫</span>
                </div>
                {selectedInvoice.discount > 0 && (
                  <div className="flex justify-between text-green-600">
                    <span>Giảm giá:</span>
                    <span className="font-semibold">-{selectedInvoice.discount.toLocaleString('vi-VN')} ₫</span>
                  </div>
                )}
                <div className="border-t pt-2 flex justify-between text-lg font-bold text-gray-900">
                  <span>Tổng cộng:</span>
                  <span>{selectedInvoice.total.toLocaleString('vi-VN')} ₫</span>
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => handlePrint(selectedInvoice)}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center justify-center gap-2"
              >
                <Printer className="w-5 h-5" /> In
              </button>
              <button
                onClick={() => {
                  setShowDetailModal(false);
                  setSelectedInvoice(null);
                }}
                className="flex-1 px-4 py-2 border rounded-lg hover:bg-gray-50"
              >
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Invoices Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-gray-500">Đang tải...</div>
        ) : filteredInvoices.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Số Hóa Đơn</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Bệnh Nhân</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Bác Sĩ</th>
                  <th className="px-6 py-3 text-right text-sm font-semibold text-gray-900">Tổng Cộng</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Trạng Thái</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Thao Tác</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {filteredInvoices.map((invoice) => (
                  <tr key={invoice.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 text-sm font-medium text-blue-600">{invoice.invoice_no}</td>
                    <td className="px-6 py-4 text-sm">
                      <div className="font-medium text-gray-900">{invoice.patient.user.full_name}</div>
                      <div className="text-xs text-gray-500">{invoice.patient.user.phone}</div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900">{invoice.appointment.doctor.user.full_name}</td>
                    <td className="px-6 py-4 text-sm text-right font-semibold text-gray-900">
                      {invoice.total.toLocaleString('vi-VN')} ₫
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(invoice.status)}`}>
                        {getStatusLabel(invoice.status)}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm flex gap-2">
                      <button
                        onClick={() => {
                          setSelectedInvoice(invoice);
                          setShowDetailModal(true);
                        }}
                        className="px-3 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 flex items-center gap-1"
                      >
                        <Eye className="w-4 h-4" /> Chi tiết
                      </button>
                      {invoice.status === 'UNPAID' && (
                        <button
                          onClick={() => handlePayment(invoice.id)}
                          className="px-3 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700 flex items-center gap-1"
                        >
                          <CheckCircle className="w-4 h-4" /> Thanh toán
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center text-gray-500">Không có hóa đơn nào</div>
        )}
      </div>
      {showCreateFromPrescriptionModal && (
        <CreateInvoiceFromPrescriptionModal
          initialPrescriptionId={modalPrescriptionId}
          onClose={() => { setShowCreateFromPrescriptionModal(false); setModalPrescriptionId(null); }}
          onInvoiceCreated={(inv) => { alert('Tạo hoá đơn thành công'); setShowCreateFromPrescriptionModal(false); setModalPrescriptionId(null); window.location.reload(); }}
        />
      )}
    </div>
  );
}
