/* eslint-disable no-console */
// frontend/src/components/admin/MedicineManagement/MedicineForm.jsx
import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { medicineService } from '../../../services/medicineService';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Modal from '../../common/Modal';

export default function MedicineForm({ isOpen, onClose, medicine, onSuccess }) {
  const [formData, setFormData] = useState({
    code: '',
    name: '',
    active_ingredient: '',
    category: '',
    unit: '',
    price: '',
    dosage_form: '',
    manufacturer: '',
    country_of_origin: '',
    description: '',
    usage_instructions: '',
    side_effects: '',
    contraindications: '',
    storage_conditions: '',
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (medicine) {
      setFormData({
        code: medicine.code || '',
        name: medicine.name || '',
        active_ingredient: medicine.active_ingredient || '',
        category: medicine.category || '',
        unit: medicine.unit || '',
        price: medicine.price || '',
        dosage_form: medicine.dosage_form || '',
        manufacturer: medicine.manufacturer || '',
        country_of_origin: medicine.country_of_origin || '',
        description: medicine.description || '',
        usage_instructions: medicine.usage_instructions || '',
        side_effects: medicine.side_effects || '',
        contraindications: medicine.contraindications || '',
        storage_conditions: medicine.storage_conditions || '',
      });
    } else {
      resetForm();
    }
  }, [medicine]);

  const resetForm = () => {
    setFormData({
      code: '',
      name: '',
      active_ingredient: '',
      category: '',
      unit: '',
      price: '',
      dosage_form: '',
      manufacturer: '',
      country_of_origin: '',
      description: '',
      usage_instructions: '',
      side_effects: '',
      contraindications: '',
      storage_conditions: '',
    });
    setErrors({});
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.code.trim()) {
      newErrors.code = 'Vui lòng nhập mã thuốc';
    }

    if (!formData.name.trim()) {
      newErrors.name = 'Vui lòng nhập tên thuốc';
    }

    if (!formData.category) {
      newErrors.category = 'Vui lòng chọn loại thuốc';
    }

    if (!formData.unit.trim()) {
      newErrors.unit = 'Vui lòng nhập đơn vị';
    }

    if (!formData.price || formData.price <= 0) {
      newErrors.price = 'Vui lòng nhập giá hợp lệ';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    try {
      setLoading(true);

      if (medicine) {
        await medicineService.updateMedicine(medicine.id, formData);
      } else {
        await medicineService.createMedicine(formData);
      }

      onSuccess();
      onClose();
      resetForm();
    } catch (error) {
      console.error('Submit error:', error);
      alert('Có lỗi xảy ra. Vui lòng thử lại!');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-800">
            {medicine ? 'Chỉnh sửa Thuốc' : 'Thêm Thuốc mới'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          <div className="grid grid-cols-2 gap-6">
            {/* Thông tin cơ bản */}
            <div className="col-span-2">
              <h3 className="text-lg font-semibold mb-4">Thông tin cơ bản</h3>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Mã thuốc <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.code}
                onChange={(e) => handleChange('code', e.target.value)}
                placeholder="VD: MED001"
              />
              {errors.code && (
                <p className="text-red-500 text-sm mt-1">{errors.code}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tên thuốc <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value)}
                placeholder="Nhập tên thuốc"
              />
              {errors.name && (
                <p className="text-red-500 text-sm mt-1">{errors.name}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Hoạt chất
              </label>
              <Input
                value={formData.active_ingredient}
                onChange={(e) => handleChange('active_ingredient', e.target.value)}
                placeholder="Nhập hoạt chất chính"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Loại thuốc <span className="text-red-500">*</span>
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                value={formData.category}
                onChange={(e) => handleChange('category', e.target.value)}
              >
                <option value="">Chọn loại thuốc</option>
                <option value="Kháng sinh">Kháng sinh</option>
                <option value="Giảm đau">Giảm đau</option>
                <option value="Hạ sốt">Hạ sốt</option>
                <option value="Vitamin">Vitamin & Khoáng chất</option>
                <option value="Tiêu hóa">Tiêu hóa</option>
                <option value="Tim mạch">Tim mạch</option>
                <option value="Hô hấp">Hô hấp</option>
                <option value="Da liễu">Da liễu</option>
                <option value="Thần kinh">Thần kinh</option>
                <option value="Khác">Khác</option>
              </select>
              {errors.category && (
                <p className="text-red-500 text-sm mt-1">{errors.category}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Dạng bào chế
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                value={formData.dosage_form}
                onChange={(e) => handleChange('dosage_form', e.target.value)}
              >
                <option value="">Chọn dạng bào chế</option>
                <option value="Viên nén">Viên nén</option>
                <option value="Viên nang">Viên nang</option>
                <option value="Siro">Siro</option>
                <option value="Thuốc tiêm">Thuốc tiêm</option>
                <option value="Thuốc nhỏ mắt">Thuốc nhỏ mắt</option>
                <option value="Kem bôi">Kem bôi</option>
                <option value="Thuốc xịt">Thuốc xịt</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Đơn vị tính <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.unit}
                onChange={(e) => handleChange('unit', e.target.value)}
                placeholder="VD: Viên, Hộp, Chai, Lọ"
              />
              {errors.unit && (
                <p className="text-red-500 text-sm mt-1">{errors.unit}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Giá bán (VNĐ) <span className="text-red-500">*</span>
              </label>
              <Input
                type="number"
                value={formData.price}
                onChange={(e) => handleChange('price', e.target.value)}
                placeholder="10000"
              />
              {errors.price && (
                <p className="text-red-500 text-sm mt-1">{errors.price}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Nhà sản xuất
              </label>
              <Input
                value={formData.manufacturer}
                onChange={(e) => handleChange('manufacturer', e.target.value)}
                placeholder="Tên nhà sản xuất"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Xuất xứ
              </label>
              <Input
                value={formData.country_of_origin}
                onChange={(e) => handleChange('country_of_origin', e.target.value)}
                placeholder="VD: Việt Nam, Ấn Độ, ..."
              />
            </div>

            {/* Thông tin chi tiết */}
            <div className="col-span-2 mt-4">
              <h3 className="text-lg font-semibold mb-4">Thông tin chi tiết</h3>
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Mô tả
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="3"
                value={formData.description}
                onChange={(e) => handleChange('description', e.target.value)}
                placeholder="Mô tả ngắn gọn về thuốc..."
              />
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Hướng dẫn sử dụng
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="3"
                value={formData.usage_instructions}
                onChange={(e) => handleChange('usage_instructions', e.target.value)}
                placeholder="Cách dùng, liều lượng..."
              />
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tác dụng phụ
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="3"
                value={formData.side_effects}
                onChange={(e) => handleChange('side_effects', e.target.value)}
                placeholder="Các tác dụng phụ có thể gặp..."
              />
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Chống chỉ định
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="2"
                value={formData.contraindications}
                onChange={(e) => handleChange('contraindications', e.target.value)}
                placeholder="Những trường hợp không nên sử dụng..."
              />
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Điều kiện bảo quản
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="2"
                value={formData.storage_conditions}
                onChange={(e) => handleChange('storage_conditions', e.target.value)}
                placeholder="Nhiệt độ, độ ẩm, ánh sáng..."
              />
            </div>
          </div>
        </form>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t bg-gray-50">
          <Button onClick={onClose} disabled={loading}>
            Hủy
          </Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Đang xử lý...' : medicine ? 'Cập nhật' : 'Thêm mới'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}