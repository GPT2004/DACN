/* eslint-disable no-unused-vars */
import React, { useEffect, useState } from 'react';
import PublicHeader from '../../components/common/PublicHeader';
import PatientPageHeader from '../../components/patient/PatientPageHeader';
import Footer from '../../components/patient/clinic/Footer';
import api from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { DollarSign, Eye, Download, RotateCw, AlertCircle } from 'lucide-react';
import InvoiceDetailModal from '../../components/patient/InvoiceDetailModal';

export default function InvoicesDetailPage() {
  useAuth();
  const [invoices, setInvoices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedInvoice, setSelectedInvoice] = useState(null);
  const [filter, setFilter] = useState('ALL'); // ALL, UNPAID, PAID, CANCELLED

  const fetchInvoices = async () => {
    try {
      setLoading(true);
      const response = await api.get('/invoices/by-patient');
      const invoiceList = response?.data?.invoices || response?.data || [];
      setInvoices(Array.isArray(invoiceList) ? invoiceList : []);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('Lỗi khi tải hóa đơn:', err);
      setInvoices([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchInvoices();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const formatDate = (date) => {
    if (!date) return 'N/A';
    return new Date(date).toLocaleDateString('vi-VN');
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('vi-VN', {
      style: 'currency',
      currency: 'VND',
      minimumFractionDigits: 0
    }).format(amount || 0);
  };

  const getStatusLabel = (status) => {
    const statusMap = {
      UNPAID: 'Chưa thanh toán',
      PAID: 'Đã thanh toán',
      CANCELLED: 'Đã hủy',
      REFUNDED: 'Đã hoàn tiền'
    };
    return statusMap[status] || status;
  };

  const getStatusColor = (status) => {
    const colorMap = {
      UNPAID: 'bg-yellow-100 text-yellow-800 border border-yellow-300',
      PAID: 'bg-green-100 text-green-800 border border-green-300',
      CANCELLED: 'bg-red-100 text-red-800 border border-red-300',
      REFUNDED: 'bg-blue-100 text-blue-800 border border-blue-300'
    };
    return colorMap[status] || 'bg-gray-100 text-gray-800';
  };

  const filteredInvoices = filter === 'ALL'
    ? invoices
    : invoices.filter(inv => inv.status === filter);

  // Group by month
  const groupedByMonth = filteredInvoices.reduce((acc, inv) => {
    const date = new Date(inv.created_at);
    const key = date.toLocaleDateString('vi-VN', { year: 'numeric', month: 'long' });
    if (!acc[key]) acc[key] = [];
    acc[key].push(inv);
    return acc;
  }, {});

  const monthKeys = Object.keys(groupedByMonth).sort((a, b) => {
    const dateA = new Date(groupedByMonth[a][0].created_at);
    const dateB = new Date(groupedByMonth[b][0].created_at);
    return dateB - dateA;
  });

  // Calculate totals
  const totals = {
    all: invoices.reduce((sum, inv) => sum + (inv.total || 0), 0),
    paid: invoices.filter(inv => inv.status === 'PAID').reduce((sum, inv) => sum + (inv.total || 0), 0),
    unpaid: invoices.filter(inv => inv.status === 'UNPAID').reduce((sum, inv) => sum + (inv.total || 0), 0)
  };

  return (
    <>
      <PublicHeader />
      <PatientPageHeader title="Quản lý hóa đơn" />

      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
        <div className="max-w-6xl mx-auto px-4 py-8">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-white rounded-lg shadow p-6 border-t-4 border-blue-500">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-600 text-sm">Tổng hóa đơn</p>
                  <p className="text-2xl font-bold text-gray-900 mt-1">{invoices.length}</p>
                </div>
                <DollarSign className="w-12 h-12 text-blue-200" />
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6 border-t-4 border-green-500">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-600 text-sm">Đã thanh toán</p>
                  <p className="text-2xl font-bold text-green-600 mt-1">{formatCurrency(totals.paid)}</p>
                </div>
                <DollarSign className="w-12 h-12 text-green-200" />
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6 border-t-4 border-yellow-500">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-600 text-sm">Chưa thanh toán</p>
                  <p className="text-2xl font-bold text-yellow-600 mt-1">{formatCurrency(totals.unpaid)}</p>
                </div>
                <AlertCircle className="w-12 h-12 text-yellow-200" />
              </div>
            </div>
          </div>

          {/* Filter and Refresh */}
          <div className="mb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="flex gap-2 flex-wrap">
              {['ALL', 'UNPAID', 'PAID', 'CANCELLED'].map((status) => (
                <button
                  key={status}
                  onClick={() => setFilter(status)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    filter === status
                      ? 'bg-blue-500 text-white shadow-lg'
                      : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {status === 'ALL' ? 'Tất cả' : getStatusLabel(status)}
                </button>
              ))}
            </div>

            <button
              onClick={fetchInvoices}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
            >
              <RotateCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Tải lại
            </button>
          </div>

          {/* Content */}
          {loading ? (
            <div className="bg-white rounded-lg shadow p-8 text-center">
              <p className="text-gray-600">Đang tải hóa đơn...</p>
            </div>
          ) : filteredInvoices.length === 0 ? (
            <div className="bg-white rounded-lg shadow p-12 text-center">
              <DollarSign className="w-16 h-16 mx-auto text-gray-300 mb-4" />
              <h3 className="text-xl font-semibold text-gray-700 mb-2">
                {filter === 'ALL' ? 'Chưa có hóa đơn' : `Không có hóa đơn ${getStatusLabel(filter).toLowerCase()}`}
              </h3>
              <p className="text-gray-500">
                {filter === 'ALL'
                  ? 'Các hóa đơn của bạn sẽ hiển thị ở đây sau khi hoàn thành dịch vụ.'
                  : 'Không có hóa đơn nào với trạng thái này.'}
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {monthKeys.map((monthKey) => (
                <div key={monthKey} className="bg-white rounded-lg shadow overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-500 to-indigo-600 px-6 py-3 text-white font-semibold">
                    {monthKey}
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-gray-50 border-b">
                        <tr>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">Mã hóa đơn</th>
                          <th className="px-6 py-3 text-left text-sm font-semibold text-gray-700">Ngày</th>
                          <th className="px-6 py-3 text-right text-sm font-semibold text-gray-700">Số tiền</th>
                          <th className="px-6 py-3 text-center text-sm font-semibold text-gray-700">Trạng thái</th>
                          <th className="px-6 py-3 text-center text-sm font-semibold text-gray-700">Thao tác</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y">
                        {groupedByMonth[monthKey].map((invoice) => (
                          <tr key={invoice.id} className="hover:bg-gray-50 transition-colors">
                            <td className="px-6 py-4 text-sm font-medium text-blue-600">#{invoice.id}</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{formatDate(invoice.created_at)}</td>
                            <td className="px-6 py-4 text-sm font-semibold text-gray-900 text-right">
                              {formatCurrency(invoice.total)}
                            </td>
                            <td className="px-6 py-4 text-center">
                              <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(invoice.status)}`}>
                                {getStatusLabel(invoice.status)}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-center">
                              <button
                                onClick={() => setSelectedInvoice(invoice)}
                                className="inline-flex items-center gap-2 px-3 py-1 text-sm text-blue-600 hover:bg-blue-50 rounded transition-colors"
                              >
                                <Eye className="w-4 h-4" />
                                Xem
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {selectedInvoice && (
        <InvoiceDetailModal
          invoice={selectedInvoice}
          onClose={() => setSelectedInvoice(null)}
        />
      )}

      <Footer />
    </>
  );
}
