import React, { useState } from 'react';
import { X, DollarSign, CreditCard, Banknote, QrCode } from 'lucide-react';
import { invoiceService } from '../../services/invoiceService';

/**
 * PaymentModal - Xử lý thanh toán hóa đơn
 */
export default function PaymentModal({ invoice, onClose, onPaymentSuccess }) {
  const [paymentMethod, setPaymentMethod] = useState('CASH');
  const items = Array.isArray(invoice?.items) ? invoice.items : [];
  const fallbackItemsSum = items.reduce((s, it) => s + (Number(it?.amount) || 0), 0);

  // try to detect a consultation/service item by common keys or text
  const consultItem = items.find((it) => {
    if (!it) return false;
    const name = String(it.service || it.description || it.label || it.name || '').toLowerCase();
    return /consult|khám|khám bệnh|consultation|phí khám|exam/.test(name);
  });

  // Determine consult fee and medicines subtotal based on items when possible
  const totalItemsSum = fallbackItemsSum;
  const consultFee = consultItem ? Number(consultItem.amount || consultItem.value || 0) : Number(invoice?.consultation_fee ?? invoice?.consultationFee ?? invoice?.doctor?.consultation_fee ?? invoice?.appointment?.doctor?.consultation_fee ?? invoice?.prescription?.doctor?.consultation_fee ?? 0);

  let medicinesSubtotal = 0;
  if (consultItem) {
    medicinesSubtotal = Math.max(0, totalItemsSum - (Number(consultItem.amount) || 0));
  } else if (invoice?.subtotal !== undefined && invoice?.subtotal !== null) {
    medicinesSubtotal = Number(invoice.subtotal || 0);
  } else {
    medicinesSubtotal = totalItemsSum;
  }

  const totalAmount = Number(invoice?.total_amount ?? invoice?.total ?? medicinesSubtotal + consultFee ?? 0);
  const paidSoFar = Number(invoice?.paid_amount ?? 0);
  const [cashGiven, setCashGiven] = useState(''); // Số tiền bệnh nhân đưa (string để nhập tự do)
  const [loading, setLoading] = useState(false);

  const remainingAmount = Math.max(0, totalAmount - paidSoFar);
  const change = Math.max(0, Number(cashGiven || 0) - remainingAmount);

  const isPaid = invoice?.status === 'PAID' || remainingAmount === 0;
  // When invoice is marked PAID but paid_amount may be missing, display total as paid
  const displayPaid = isPaid ? (paidSoFar > 0 ? paidSoFar : totalAmount) : paidSoFar;

  const handlePayment = async () => {
    try {
      setLoading(true);
      // Call API to pay invoice. We pass the total paid amount (existing paidSoFar + cashGiven)
      const amountToPay = Number(paidSoFar || 0) + Number(cashGiven || 0);
      // map UI method to backend-expected values
      let apiPaymentMethod = paymentMethod;
      if (apiPaymentMethod === 'QR') apiPaymentMethod = 'transfer';
      apiPaymentMethod = String(apiPaymentMethod || '').toLowerCase();
      const res = await invoiceService.payInvoice(invoice.id, apiPaymentMethod, amountToPay);
      const data = res.data || res;
      // data expected: { invoice, change }
      onPaymentSuccess({
        invoiceId: invoice.id,
        paidAmount: amountToPay,
        paymentMethod,
        change: data.change ?? change,
        invoice: data.invoice || data,
        timestamp: new Date().toISOString()
      });
      onClose();
    } catch (error) {
      alert('Lỗi xử lý thanh toán: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full">
        <div className="bg-gradient-to-r from-green-50 to-green-100 p-6 border-b flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
            <DollarSign size={24} /> Xử lý thanh toán
          </h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700">
            <X size={24} />
          </button>
        </div>

        <div className="p-6 space-y-4">
          {/* Invoice Summary */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Hóa đơn:</span>
                <span className="font-semibold">#{invoice?.invoice_number ?? invoice?.id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Bệnh nhân:</span>
                <span className="font-semibold">{invoice?.patient?.user?.full_name}</span>
              </div>
              <div className="border-b pb-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Phí khám:</span>
                  <span className="font-semibold">{consultFee.toLocaleString('vi-VN')} VND</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Thuốc:</span>
                  <span className="font-semibold">{medicinesSubtotal.toLocaleString('vi-VN')} VND</span>
                </div>
              </div>
              <div className="border-t pt-2 flex justify-between">
                <span className="text-gray-600">Tổng tiền:</span>
                <span className="font-bold text-lg">{totalAmount.toLocaleString('vi-VN')} VND</span>
              </div>
              {!isPaid && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Đã thanh toán:</span>
                  <span className="font-semibold">{paidSoFar.toLocaleString('vi-VN')} VND</span>
                </div>
              )}
              {!isPaid ? (
                <div className="bg-orange-100 p-2 rounded flex justify-between">
                  <span className="text-orange-800 font-semibold">Còn nợ:</span>
                  <span className="text-orange-800 font-bold">{Math.max(0, remainingAmount).toLocaleString('vi-VN')} VND</span>
                </div>
              ) : (
                <div>
                  <div className="bg-green-100 p-2 rounded flex justify-between items-center">
                    <span className="text-green-800 font-semibold">Đã thanh toán</span>
                    <span className="text-green-800 font-bold">{displayPaid.toLocaleString('vi-VN')} VND</span>
                  </div>
                  <div className="mt-2 text-sm text-gray-600 flex justify-between">
                    <span className="">Còn nợ:</span>
                    <span className="font-semibold">0 VND</span>
                  </div>
                </div>
              )}
            </div>
          </div>
          {/* Payment Method - only when unpaid */}
          {!isPaid && (
            <div>
              <label className="text-sm font-semibold text-gray-900 mb-3 block">Phương thức thanh toán</label>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { id: 'CASH', label: 'Tiền mặt', icon: Banknote },
                  { id: 'CARD', label: 'Thẻ', icon: CreditCard },
                  { id: 'QR', label: 'QR Code', icon: QrCode }
                ].map(method => {
                  const Icon = method.icon;
                  return (
                    <button
                      key={method.id}
                      onClick={() => setPaymentMethod(method.id)}
                      className={`p-3 rounded-lg border-2 transition flex flex-col items-center gap-2 ${
                        paymentMethod === method.id
                          ? 'border-green-600 bg-green-50'
                          : 'border-gray-200 hover:border-green-300'
                      }`}
                    >
                      <Icon size={20} className={paymentMethod === method.id ? 'text-green-600' : 'text-gray-600'} />
                      <span className="text-xs font-medium">{method.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Amount Input - only show when invoice is not yet paid */}
          {!isPaid && (
            <>
              <div>
                <label className="text-sm font-semibold text-gray-900 mb-2 block">Số tiền bệnh nhân đưa</label>
                <div className="relative">
                  <input
                    type="text"
                    inputMode="numeric"
                    value={cashGiven}
                    onChange={(e) => {
                      // chỉ cho phép nhập số (loại bỏ kí tự khác)
                      const cleaned = String(e.target.value).replace(/[^0-9]/g, '');
                      setCashGiven(cleaned);
                    }}
                    placeholder="Nhập số tiền bệnh nhân đưa"
                    className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-green-600"
                  />
                  <span className="absolute right-4 top-2 text-gray-600 text-sm">VND</span>
                </div>
              </div>

              {/* Tiền trả lại */}
              <div>
                <label className="text-sm font-semibold text-gray-900 mb-2 block">Tiền trả lại</label>
                <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                  <p className="text-sm text-blue-800 font-semibold">{change.toLocaleString('vi-VN')} VND</p>
                  <p className="text-xs text-gray-600">(Tự tính: Số tiền bệnh nhân đưa - Còn phải thanh toán)</p>
                </div>
              </div>
            </>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4 border-t">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 font-medium"
            >
              {isPaid ? 'Đóng' : 'Hủy'}
            </button>
            {!isPaid && (
              <button
                onClick={handlePayment}
                disabled={loading || Number(cashGiven || 0) < remainingAmount}
                className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 font-medium"
              >
                {loading ? 'Đang xử lý...' : 'Xác nhận thanh toán'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
