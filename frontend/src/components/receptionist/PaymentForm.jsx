// src/components/receptionist/PaymentForm.jsx
import React, { useState } from 'react';
import { CreditCard, Wallet, QrCode, CheckCircle } from 'lucide-react';

const PaymentForm = ({ amount, onPay, alreadyPaid = 0 }) => {
  const [method, setMethod] = useState('CASH');
  const [paidAmount, setPaidAmount] = useState(amount || 0);
  const [otherDeduction, setOtherDeduction] = useState(0);

  const methods = [
    { id: 'CASH', label: 'Tiền mặt', icon: Wallet },
    { id: 'CARD', label: 'Thẻ', icon: CreditCard },
    { id: 'QR', label: 'QR Code', icon: QrCode },
  ];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-bold text-gray-900 mb-4">Thanh toán</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Số tiền cần thanh toán</label>
          <p className="text-2xl font-bold text-green-600">{amount.toLocaleString('vi-VN')} ₫</p>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Phương thức</label>
          <div className="grid grid-cols-3 gap-3">
            {methods.map(m => (
              <button
                key={m.id}
                onClick={() => setMethod(m.id)}
                className={`p-4 border rounded-lg flex flex-col items-center gap-2 transition ${
                  method === m.id ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
                }`}
              >
                <m.icon className="w-6 h-6" />
                <span className="text-sm font-medium">{m.label}</span>
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Số tiền cần thanh toán</label>
              <p className="text-2xl font-bold text-green-600">{(amount || 0).toLocaleString('vi-VN')} ₫</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Đã thanh toán</label>
              <p className="text-lg font-semibold text-gray-700">{(alreadyPaid || 0).toLocaleString('vi-VN')} ₫</p>
            </div>
          </div>

          <label className="block text-sm font-medium text-gray-700 mb-1 mt-4">Số tiền nhận từ bệnh nhân</label>
          <input
            type="number"
            className="w-full px-3 py-2 border rounded-lg"
            value={paidAmount}
            onChange={(e) => setPaidAmount(Number(e.target.value || 0))}
          />

          <label className="block text-sm font-medium text-gray-700 mb-1 mt-3">Khấu trừ / Thanh toán khác (VND)</label>
          <input
            type="number"
            className="w-full px-3 py-2 border rounded-lg"
            value={otherDeduction}
            onChange={(e) => setOtherDeduction(Number(e.target.value || 0))}
          />

          <p className="text-sm text-gray-600 mt-2">
            Tiền thừa: <span className="font-medium text-green-600">
              {Math.max(0, (paidAmount + otherDeduction + (alreadyPaid || 0)) - (amount || 0)).toLocaleString('vi-VN')} ₫
            </span>
          </p>
          <p className="text-sm text-gray-600 mt-1">
            Còn nợ: <span className="font-medium text-red-600">
              {Math.max(0, (amount || 0) - (paidAmount + otherDeduction + (alreadyPaid || 0))).toLocaleString('vi-VN')} ₫
            </span>
          </p>
        </div>
        <button
          onClick={() => {
            const netPaid = Number(paidAmount || 0) + Number(otherDeduction || 0) + Number(alreadyPaid || 0);
            const change = Math.max(0, netPaid - (amount || 0));
            onPay({ method, paidAmount: Number(paidAmount || 0), other: Number(otherDeduction || 0), netPaid, change });
          }}
          className="w-full py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center justify-center gap-2"
        >
          <CheckCircle className="w-5 h-5" />
          Xác nhận thanh toán
        </button>
      </div>
    </div>
  );
};

export default PaymentForm;