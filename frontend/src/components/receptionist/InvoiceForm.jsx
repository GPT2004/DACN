// src/components/receptionist/InvoiceForm.jsx
import React, { useState } from 'react';
import { FileText, Plus, Trash2, Printer } from 'lucide-react';

const InvoiceForm = ({ patient, onSave }) => {
  const [items, setItems] = useState([
    { id: 1, name: 'Phí khám', price: 150000, qty: 1 },
  ]);

  const total = items.reduce((sum, item) => sum + item.price * item.qty, 0);

  const addItem = () => {
    setItems([...items, { id: Date.now(), name: '', price: 0, qty: 1 }]);
  };

  const removeItem = (id) => {
    setItems(items.filter(i => i.id !== id));
  };

  const updateItem = (id, field, value) => {
    setItems(items.map(i => i.id === id ? { ...i, [field]: value } : i));
  };

  return (
    <div className="bg-white rounded-lg shadow-lg">
      <div className="p-6 border-b">
        <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <FileText className="w-6 h-6" />
          Tạo hóa đơn
        </h3>
        <p className="text-sm text-gray-600">Bệnh nhân: {patient?.name}</p>
      </div>
      <div className="p-6 space-y-4">
        {items.map((item, idx) => (
          <div key={item.id} className="flex items-center gap-3">
            <input
              type="text"
              placeholder="Tên dịch vụ"
              className="flex-1 px-3 py-2 border rounded-lg"
              value={item.name}
              onChange={(e) => updateItem(item.id, 'name', e.target.value)}
            />
            <input
              type="number"
              placeholder="Giá"
              className="w-32 px-3 py-2 border rounded-lg"
              value={item.price}
              onChange={(e) => updateItem(item.id, 'price', +e.target.value)}
            />
            <input
              type="number"
              placeholder="SL"
              className="w-20 px-3 py-2 border rounded-lg"
              value={item.qty}
              onChange={(e) => updateItem(item.id, 'qty', +e.target.value)}
            />
            <button
              onClick={() => removeItem(item.id)}
              className="p-2 text-red-600 hover:bg-red-50 rounded"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        ))}
        <button
          onClick={addItem}
          className="w-full py-2 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-blue-500 hover:text-blue-600 flex items-center justify-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Thêm dịch vụ
        </button>
        <div className="border-t pt-4">
          <div className="flex justify-between text-lg font-bold">
            <span>Tổng cộng:</span>
            <span>{total.toLocaleString('vi-VN')} ₫</span>
          </div>
        </div>
        <div className="flex gap-3">
          <button className="flex-1 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center justify-center gap-2">
            <Printer className="w-4 h-4" />
            In hóa đơn
          </button>
          <button
            onClick={() => onSave({ items, total })}
            className="flex-1 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            Lưu
          </button>
        </div>
      </div>
    </div>
  );
};

export default InvoiceForm;