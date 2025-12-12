// src/components/receptionist/PatientRegistrationForm.jsx
import React, { useState } from 'react';
import { User, Phone, Mail, Home, Calendar, Save } from 'lucide-react';

const PatientRegistrationForm = ({ onSave, onCancel }) => {
  const [formData, setFormData] = useState({
    name: '', phone: '', email: '', gender: 'Nam', dob: '', address: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-bold text-gray-900 mb-6">Đăng ký bệnh nhân mới</h3>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              <User className="w-4 h-4 inline mr-1" />
              Họ tên *
            </label>
            <input
              type="text"
              required
              className="w-full px-3 py-2 border rounded-lg"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              <Phone className="w-4 h-4 inline mr-1" />
              Số điện thoại *
            </label>
            <input
              type="tel"
              required
              className="w-full px-3 py-2 border rounded-lg"
              value={formData.phone}
              onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              <Mail className="w-4 h-4 inline mr-1" />
              Email
            </label>
            <input
              type="email"
              className="w-full px-3 py-2 border rounded-lg"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Giới tính
            </label>
            <select
              className="w-full px-3 py-2 border rounded-lg"
              value={formData.gender}
              onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
            >
              <option>Nam</option>
              <option>Nữ</option>
              <option>Khác</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              <Calendar className="w-4 h-4 inline mr-1" />
              Ngày sinh
            </label>
            <input
              type="date"
              className="w-full px-3 py-2 border rounded-lg"
              value={formData.dob}
              onChange={(e) => setFormData({ ...formData, dob: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              <Home className="w-4 h-4 inline mr-1" />
              Địa chỉ
            </label>
            <input
              type="text"
              className="w-full px-3 py-2 border rounded-lg"
              value={formData.address}
              onChange={(e) => setFormData({ ...formData, address: e.target.value })}
            />
          </div>
        </div>
        <div className="flex justify-end gap-3 pt-4 border-t">
          <button type="button" onClick={onCancel} className="px-6 py-2 border rounded-lg hover:bg-gray-50">
            Hủy
          </button>
          <button type="submit" className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
            <Save className="w-4 h-4" />
            Lưu bệnh nhân
          </button>
        </div>
      </form>
    </div>
  );
};

export default PatientRegistrationForm;