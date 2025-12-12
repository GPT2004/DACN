import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { createDoctor } from '../../services/doctorService';
import axios from 'axios';

export default function AddDoctorModal({ onClose, onSuccess }) {
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    phone: '',
    specialty: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [specialties, setSpecialties] = useState([]);

  useEffect(() => {
    fetchSpecialties();
  }, []);

  const fetchSpecialties = async () => {
    try {
      const response = await axios.get('/api/doctors/public/specialties/list');
      setSpecialties(response.data || []);
    } catch (error) {
      console.error('Error fetching specialties:', error);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.full_name.trim() || !formData.email.trim()) {
      setError('Vui lòng điền tên và email');
      return;
    }

    try {
      setLoading(true);
      await createDoctor(formData);
      onSuccess();
      onClose();
    } catch (err) {
      setError(err.response?.data?.message || 'Tạo bác sĩ thất bại');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 max-h-96 overflow-y-auto">
        <div className="sticky top-0 flex items-center justify-between p-6 border-b border-gray-200 bg-white">
          <h2 className="text-xl font-bold text-gray-900">Thêm bác sĩ mới</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 transition"><X size={24} /></button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && <div className="text-sm text-red-600">{error}</div>}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Họ và tên</label>
            <input name="full_name" value={formData.full_name} onChange={handleChange} className="w-full px-3 py-2 border rounded" />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input name="email" type="email" value={formData.email} onChange={handleChange} className="w-full px-3 py-2 border rounded" />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Số điện thoại</label>
            <input name="phone" value={formData.phone} onChange={handleChange} className="w-full px-3 py-2 border rounded" />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Chuyên khoa</label>
            <select name="specialty" value={formData.specialty} onChange={handleChange} className="w-full px-3 py-2 border rounded">
              <option value="">Chọn chuyên khoa</option>
              {specialties.map(spec => (
                <option key={spec.id} value={spec.name}>{spec.name}</option>
              ))}
            </select>
          </div>

          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg">Hủy</button>
            <button type="submit" disabled={loading} className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg">{loading ? 'Đang tạo...' : 'Tạo'}</button>
          </div>
        </form>
      </div>
    </div>
  );
}
