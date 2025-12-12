import React, { useState } from 'react';
import { X, DollarSign, Package, Download } from 'lucide-react';

export default function InvoiceDetailModal({ invoice, onClose }) {
  const [printing, setPrinting] = useState(false);

  if (!invoice) return null;

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

  const items = Array.isArray(invoice.items) ? invoice.items : [];

  const handlePrint = () => {
    setPrinting(true);
    window.print();
    setTimeout(() => setPrinting(false), 1000);
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b p-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Chi tiết hóa đơn</h2>
            <p className="text-sm text-gray-500">Hóa đơn #{invoice.id}</p>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Status and Date */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Trạng thái</p>
              <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium mt-1 ${getStatusColor(invoice.status)}`}>
                {getStatusLabel(invoice.status)}
              </span>
            </div>
            <div>
              <p className="text-sm text-gray-600">Ngày phát hành</p>
              <p className="font-semibold text-gray-900">{formatDate(invoice.created_at)}</p>
            </div>
          </div>

          {/* Appointment Info */}
          {invoice.appointment && (
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Thông tin lịch hẹn</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Ngày khám:</span>
                  <span className="font-medium text-gray-900">
                    {formatDate(invoice.appointment.appointment_date)}
                  </span>
                </div>
                {invoice.appointment.reason && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Lý do khám:</span>
                    <span className="font-medium text-gray-900">{invoice.appointment.reason}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Items */}
          <div className="border-t pt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Package className="w-5 h-5" />
              Các dịch vụ
            </h3>
            {items.length > 0 ? (
              <div className="space-y-3">
                {items.map((item, idx) => (
                  <div key={idx} className="flex justify-between items-start pb-3 border-b border-gray-200 last:border-0">
                    <div className="flex-1">
                      <p className="font-medium text-gray-800">{item.name || item.description || `Dịch vụ ${idx + 1}`}</p>
                      {item.quantity && (
                        <p className="text-sm text-gray-600">
                          Số lượng: {item.quantity} {item.unit || ''}
                        </p>
                      )}
                    </div>
                    <div className="text-right">
                      {item.unit_price && (
                        <p className="text-sm text-gray-600">
                          Đơn giá: {formatCurrency(item.unit_price)}
                        </p>
                      )}
                      <p className="font-semibold text-gray-900">
                        {formatCurrency(item.total || item.amount || 0)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-600">Không có dịch vụ nào</p>
            )}
          </div>

          {/* Summary */}
          <div className="border-t pt-6 bg-gray-50 p-4 rounded-lg space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-700">Tạm tính:</span>
              <span className="font-semibold text-gray-900">{formatCurrency(invoice.subtotal)}</span>
            </div>
            {invoice.tax > 0 && (
              <div className="flex justify-between">
                <span className="text-gray-700">Thuế:</span>
                <span className="font-semibold text-gray-900">{formatCurrency(invoice.tax)}</span>
              </div>
            )}
            {invoice.discount > 0 && (
              <div className="flex justify-between text-green-600">
                <span>Giảm giá:</span>
                <span className="font-semibold">-{formatCurrency(invoice.discount)}</span>
              </div>
            )}
            <div className="border-t border-gray-300 pt-3 flex justify-between items-center">
              <span className="text-lg font-semibold text-gray-900">Tổng cộng:</span>
              <span className="text-2xl font-bold text-blue-600 flex items-center gap-2">
                <DollarSign className="w-6 h-6" />
                {formatCurrency(invoice.total)}
              </span>
            </div>
            {invoice.paid_at && (
              <div className="text-sm text-green-700 bg-green-100 p-2 rounded border border-green-300">
                Thanh toán lúc: {new Date(invoice.paid_at).toLocaleString('vi-VN')}
              </div>
            )}
          </div>

          {/* Notes */}
          {invoice.notes && (
            <div className="border-t pt-6 bg-blue-50 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-900 mb-2">Ghi chú</h4>
              <p className="text-blue-800">{invoice.notes}</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-white border-t p-6 flex gap-3 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
          >
            Đóng
          </button>
          <button
            onClick={handlePrint}
            disabled={printing}
            className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            {printing ? 'Đang in...' : 'In hóa đơn'}
          </button>
        </div>
      </div>
    </div>
  );
}
