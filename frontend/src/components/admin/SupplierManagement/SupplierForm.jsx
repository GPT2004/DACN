// frontend/src/components/admin/SupplierManagement/SupplierForm.jsx
import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { supplierService } from '../../../services/supplierService';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Modal from '../../common/Modal';

export default function SupplierForm({ isOpen, onClose, supplier, onSuccess }) {
  const [formData, setFormData] = useState({
    code: '',
    name: '',
    contact_person: '',
    phone: '',
    email: '',
    address: '',
    tax_code: '',
    bank_account: '',
    bank_name: '',
    notes: '',
    is_active: true,
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (supplier) {
      setFormData({
        code: supplier.code || '',
        name: supplier.name || '',
        contact_person: supplier.contact_person || '',
        phone: supplier.phone || '',
        email: supplier.email || '',
        address: supplier.address || '',
        tax_code: supplier.tax_code || '',
        bank_account: supplier.bank_account || '',
        bank_name: supplier.bank_name || '',
        notes: supplier.notes || '',
        is_active: supplier.is_active !== false,
      });
    } else {
      resetForm();
    }
  }, [supplier]);

  const resetForm = () => {
    setFormData({
      code: '',
      name: '',
      contact_person: '',
      phone: '',
      email: '',
      address: '',
      tax_code: '',
      bank_account: '',
      bank_name: '',
      notes: '',
      is_active: true,
    });
    setErrors({});
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Vui lòng nhập tên nhà cung cấp';
    }

    if (!formData.phone.trim()) {
      newErrors.phone = 'Vui lòng nhập số điện thoại';
    }

    if (formData.email && !/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email không hợp lệ';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    try {
      setLoading(true);

      if (supplier) {
        await supplierService.updateSupplier(supplier.id, formData);
      } else {
        await supplierService.createSupplier(formData);
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
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-2xl font-bold text-gray-800">
            {supplier ? 'Chỉnh sửa Nhà cung cấp' : 'Thêm Nhà cung cấp mới'}
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
            {/* Basic Info */}
            <div className="col-span-2">
              <h3 className="text-lg font-semibold mb-4">Thông tin cơ bản</h3>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Mã nhà cung cấp
              </label>
              <Input
                value={formData.code}
                onChange={(e) => handleChange('code', e.target.value)}
                placeholder="VD: SUP001"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tên nhà cung cấp <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value)}
                placeholder="Nhập tên nhà cung cấp"
              />
              {errors.name && (
                <p className="text-red-500 text-sm mt-1">{errors.name}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Người liên hệ
              </label>
              <Input
                value={formData.contact_person}
                onChange={(e) => handleChange('contact_person', e.target.value)}
                placeholder="Họ tên người liên hệ"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Số điện thoại <span className="text-red-500">*</span>
              </label>
              <Input
                value={formData.phone}
                onChange={(e) => handleChange('phone', e.target.value)}
                placeholder="0123456789"
              />
              {errors.phone && (
                <p className="text-red-500 text-sm mt-1">{errors.phone}</p>
              )}
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email
              </label>
              <Input
                type="email"
                value={formData.email}
                onChange={(e) => handleChange('email', e.target.value)}
                placeholder="email@example.com"
              />
              {errors.email && (
                <p className="text-red-500 text-sm mt-1">{errors.email}</p>
              )}
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Địa chỉ
              </label>
              <Input
                value={formData.address}
                onChange={(e) => handleChange('address', e.target.value)}
                placeholder="Địa chỉ công ty"
              />
            </div>

            {/* Tax & Banking Info */}
            <div className="col-span-2 mt-4">
              <h3 className="text-lg font-semibold mb-4">Thông tin thuế & ngân hàng</h3>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Mã số thuế
              </label>
              <Input
                value={formData.tax_code}
                onChange={(e) => handleChange('tax_code', e.target.value)}
                placeholder="Nhập mã số thuế"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tên ngân hàng
              </label>
              <Input
                value={formData.bank_name}
                onChange={(e) => handleChange('bank_name', e.target.value)}
                placeholder="VD: Vietcombank, BIDV"
              />
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Số tài khoản
              </label>
              <Input
                value={formData.bank_account}
                onChange={(e) => handleChange('bank_account', e.target.value)}
                placeholder="Nhập số tài khoản ngân hàng"
              />
            </div>

            {/* Additional Info */}
            <div className="col-span-2 mt-4">
              <h3 className="text-lg font-semibold mb-4">Thông tin bổ sung</h3>
            </div>

            <div className="col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Ghi chú
              </label>
              <textarea
                className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                rows="4"
                value={formData.notes}
                onChange={(e) => handleChange('notes', e.target.value)}
                placeholder="Ghi chú về nhà cung cấp..."
              />
            </div>

            <div className="col-span-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                  checked={formData.is_active}
                  onChange={(e) => handleChange('is_active', e.target.checked)}
                />
                <span className="ml-2 text-sm text-gray-700">
                  Nhà cung cấp đang hoạt động
                </span>
              </label>
            </div>
          </div>
        </form>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t bg-gray-50">
          <Button onClick={onClose} disabled={loading}>
            Hủy
          </Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? 'Đang xử lý...' : supplier ? 'Cập nhật' : 'Thêm mới'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}