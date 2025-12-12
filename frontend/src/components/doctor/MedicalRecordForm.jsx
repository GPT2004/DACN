// src/components/doctor/MedicalRecordForm.jsx
import React, { useState } from 'react';
import { Thermometer, Heart, Activity, Weight, Save } from 'lucide-react';

const MedicalRecordForm = ({ patient, onSave, onCancel }) => {
  const [formData, setFormData] = useState({
    symptoms: '',
    diagnosis: '',
    temperature: '',
    bloodPressure: '',
    heartRate: '',
    weight: '',
    height: '',
    notes: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg">
      <div className="p-6 border-b">
        <h3 className="text-xl font-bold text-gray-900">Ghi Bệnh án</h3>
        <p className="text-sm text-gray-600 mt-1">Bệnh nhân: {patient?.name}</p>
      </div>
      <form onSubmit={handleSubmit} className="p-6 space-y-6">
        <div>
          <h4 className="font-semibold text-gray-900 mb-3">Chỉ số sinh tồn</h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <Thermometer className="w-4 h-4 inline mr-1" />
                Nhiệt độ (°C)
              </label>
              <input
                type="number"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                placeholder="36.5"
                value={formData.temperature}
                onChange={(e) => setFormData({ ...formData, temperature: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <Heart className="w-4 h-4 inline mr-1" />
                Huyết áp (mmHg)
              </label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                placeholder="120/80"
                value={formData.bloodPressure}
                onChange={(e) => setFormData({ ...formData, bloodPressure: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <Activity className="w-4 h-4 inline mr-1" />
                Nhịp tim (bpm)
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                placeholder="75"
                value={formData.heartRate}
                onChange={(e) => setFormData({ ...formData, heartRate: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <Weight className="w-4 h-4 inline mr-1" />
                Cân nặng (kg)
              </label>
              <input
                type="number"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                placeholder="65.0"
                value={formData.weight}
                onChange={(e) => setFormData({ ...formData, weight: e.target.value })}
              />
            </div>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Triệu chứng</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            rows="3"
            placeholder="Mô tả triệu chứng..."
            value={formData.symptoms}
            onChange={(e) => setFormData({ ...formData, symptoms: e.target.value })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Chẩn đoán *</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            rows="3"
            placeholder="Nhập chẩn đoán..."
            value={formData.diagnosis}
            onChange={(e) => setFormData({ ...formData, diagnosis: e.target.value })}
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Ghi chú</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            rows="4"
            placeholder="Ghi chú thêm..."
            value={formData.notes}
            onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
          />
        </div>

        <div className="flex justify-end gap-3 pt-4 border-t">
          <button type="button" onClick={onCancel} className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50">
            Hủy
          </button>
          <button type="submit" className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
            <Save className="w-4 h-4" />
            Lưu bệnh án
          </button>
        </div>
      </form>
    </div>
  );
};

export default MedicalRecordForm;