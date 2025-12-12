// src/components/doctor/LabOrderForm.jsx
import React, { useState } from 'react';
import { TestTube } from 'lucide-react';

const LabOrderForm = ({ patient, onSave, onCancel }) => {
  const [selectedTests, setSelectedTests] = useState([]);
  const [notes, setNotes] = useState('');

  const availableTests = [
    { id: 1, name: 'Xét nghiệm máu tổng quát', code: 'CBC' },
    { id: 2, name: 'Xét nghiệm nước tiểu', code: 'UA' },
    { id: 3, name: 'Đường huyết', code: 'GLU' },
    { id: 4, name: 'Chức năng gan', code: 'LFT' },
    { id: 5, name: 'Chức năng thận', code: 'RFT' },
    { id: 6, name: 'X-quang phổi', code: 'CXR' },
    { id: 7, name: 'Siêu âm bụng', code: 'USG' },
    { id: 8, name: 'Điện tâm đồ', code: 'ECG' }
  ];

  const toggleTest = (testId) => {
    setSelectedTests(prev =>
      prev.includes(testId) ? prev.filter(id => id !== testId) : [...prev, testId]
    );
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const tests = availableTests.filter(t => selectedTests.includes(t.id));
    onSave({ tests, notes });
  };

  return (
    <div className="bg-white rounded-lg shadow-lg">
      <div className="p-6 border-b">
        <h3 className="text-xl font-bold text-gray-900">Chỉ định Xét nghiệm</h3>
        <p className="text-sm text-gray-600 mt-1">Bệnh nhân: {patient?.name}</p>
      </div>
      <form onSubmit={handleSubmit} className="p-6 space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">Chọn xét nghiệm *</label>
          <div className="grid grid-cols-2 gap-3">
            {availableTests.map(test => (
              <label
                key={test.id}
                className={`flex items-center gap-3 p-3 border-2 rounded-lg cursor-pointer transition ${
                  selectedTests.includes(test.id)
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedTests.includes(test.id)}
                  onChange={() => toggleTest(test.id)}
                  className="w-4 h-4 text-blue-600"
                />
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{test.name}</div>
                  <div className="text-xs text-gray-500">{test.code}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Ghi chú</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            rows="3"
            placeholder="Ghi chú thêm..."
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </div>
        {selectedTests.length > 0 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-medium text-blue-900 mb-2">Đã chọn {selectedTests.length} xét nghiệm</h4>
            <div className="space-y-1">
              {availableTests
                .filter(t => selectedTests.includes(t.id))
                .map(test => (
                  <div key={test.id} className="text-sm text-blue-700">• {test.name}</div>
                ))}
            </div>
          </div>
        )}
        <div className="flex justify-end gap-3 pt-4 border-t">
          <button type="button" onClick={onCancel} className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50">
            Hủy
          </button>
          <button
            type="submit"
            disabled={selectedTests.length === 0}
            className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 flex items-center gap-2"
          >
            <TestTube className="w-4 h-4" />
            Lưu chỉ định
          </button>
        </div>
      </form>
    </div>
  );
};

export default LabOrderForm;