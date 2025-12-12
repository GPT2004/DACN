import React, { useState } from 'react';
import { Users, AlertCircle, CheckCircle } from 'lucide-react';
import PatientRegistrationForm from '../../components/receptionist/PatientRegistrationForm';

export default function PatientRegistrationPage() {
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [createdPatient, setCreatedPatient] = useState(null);

  const handleSave = (formData) => {
    (async () => {
      try {
        // Validate basic required fields
        if (!formData.name || !formData.phone) {
          setErrorMessage('Vui lòng nhập đầy đủ thông tin bắt buộc');
          return;
        }

        setErrorMessage('');
        // Build payload matching backend createPatient schema
        const mapGender = (g) => {
          if (!g) return undefined;
          const v = String(g).toLowerCase();
          if (v === 'nam' || v === 'male') return 'Male';
          if (v === 'nữ' || v === 'nu' || v === 'female') return 'Female';
          return 'Other';
        };

        const email = formData.email && formData.email.trim().length > 0
          ? formData.email.trim()
          : `patient+${Date.now()}@example.test`;

        const password = `p@tient${Math.floor(Math.random() * 90000) + 10000}`;

        const payload = {
          email,
          password,
          full_name: formData.name,
          phone: formData.phone,
          dob: formData.dob || undefined,
          gender: mapGender(formData.gender),
        };

        // call API
        setSuccessMessage('');
        const res = await (await import('../../services/patientService')).patientService.createPatient(payload);

        const created = res?.data?.data || res?.data || null;
        if (!created) {
          throw new Error('Unexpected response from server');
        }

        // Map created patient to the UI shape
        const newPatient = {
          id: created.id || created.user_id || Math.floor(Math.random() * 100000),
          name: created.full_name || formData.name,
          phone: created.phone || formData.phone,
          email: created.email || email,
          dob: created.dob || formData.dob,
          gender: formData.gender,
          address: formData.address || '',
          created_at: created.created_at || new Date().toISOString(),
        };

        setCreatedPatient(newPatient);
        setSuccessMessage(`Đăng ký thành công! Mã bệnh nhân: BN${newPatient.id}`);

        // Clear success after 3s
        setTimeout(() => setSuccessMessage(''), 3000);
      } catch (error) {
        console.error('Error saving patient:', error?.response || error);
        const message = error?.response?.data?.message || 'Lỗi khi đăng ký bệnh nhân';
        setErrorMessage(message);
        setSuccessMessage('');
      }
    })();
  };

  const handleCancel = () => {
    setCreatedPatient(null);
    setSuccessMessage('');
    setErrorMessage('');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
          <Users className="w-8 h-8 text-blue-600" />
          Đăng Ký Bệnh Nhân Mới
        </h1>
        <p className="text-gray-600 mt-1">Thêm bệnh nhân mới vào hệ thống</p>
      </div>

      {/* Messages */}
      {successMessage && (
        <div className="p-4 bg-green-50 border border-green-200 rounded-lg flex items-start gap-3">
          <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-green-900">{successMessage}</p>
            {createdPatient && (
              <p className="text-sm text-green-700 mt-1">
                Có thể bắt đầu đặt lịch khám cho bệnh nhân này
              </p>
            )}
          </div>
        </div>
      )}

      {errorMessage && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-red-900">{errorMessage}</p>
          </div>
        </div>
      )}

      {/* Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <PatientRegistrationForm
          onSave={handleSave}
          onCancel={handleCancel}
        />
      </div>

      {/* Created Patient Info */}
      {createdPatient && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-blue-900 mb-4">Thông Tin Bệnh Nhân Đã Tạo</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-blue-700">Mã Bệnh Nhân</p>
              <p className="text-xl font-bold text-blue-900">BN{createdPatient.id}</p>
            </div>
            <div>
              <p className="text-sm text-blue-700">Họ Tên</p>
              <p className="text-xl font-bold text-blue-900">{createdPatient.name}</p>
            </div>
            <div>
              <p className="text-sm text-blue-700">Số Điện Thoại</p>
              <p className="text-lg font-semibold text-blue-900">{createdPatient.phone}</p>
            </div>
            <div>
              <p className="text-sm text-blue-700">Email</p>
              <p className="text-lg font-semibold text-blue-900">{createdPatient.email || '-'}</p>
            </div>
            <div>
              <p className="text-sm text-blue-700">Giới Tính</p>
              <p className="text-lg font-semibold text-blue-900">{createdPatient.gender}</p>
            </div>
            <div>
              <p className="text-sm text-blue-700">Ngày Sinh</p>
              <p className="text-lg font-semibold text-blue-900">
                {createdPatient.dob ? new Date(createdPatient.dob).toLocaleDateString('vi-VN') : '-'}
              </p>
            </div>
            <div className="md:col-span-2">
              <p className="text-sm text-blue-700">Địa Chỉ</p>
              <p className="text-lg font-semibold text-blue-900">{createdPatient.address || '-'}</p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-100 rounded border border-blue-300">
            <p className="text-sm text-blue-900">
              ✅ Bệnh nhân đã được thêm vào hệ thống. Bạn có thể:
            </p>
            <ul className="list-disc list-inside text-sm text-blue-900 mt-2 space-y-1">
              <li>Đặt lịch khám cho bệnh nhân</li>
              <li>Cập nhật thông tin khi cần thiết</li>
              <li>Theo dõi lịch sử khám bệnh</li>
              <li>In thẻ bệnh nhân nếu cần</li>
            </ul>
          </div>

          <div className="mt-4 flex gap-3">
            <button
              onClick={() => {
                setCreatedPatient(null);
                setSuccessMessage('');
              }}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
            >
              Đăng Ký Bệnh Nhân Khác
            </button>
            <button
              onClick={() => window.location.href = '/receptionist/appointments'}
              className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
            >
              Đặt Lịch Khám
            </button>
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <h3 className="font-semibold text-yellow-900 mb-3">💡 Hướng Dẫn Đăng Ký</h3>
        <ul className="space-y-2 text-sm text-yellow-800">
          <li>• <strong>Thông tin bắt buộc:</strong> Họ tên, Số điện thoại</li>
          <li>• <strong>Số điện thoại:</strong> Sử dụng để liên lạc với bệnh nhân</li>
          <li>• <strong>Email:</strong> Tùy chọn, dùng để gửi thông báo</li>
          <li>• <strong>Ngày sinh:</strong> Giúp tính toán độ tuổi và liều lượng thuốc</li>
          <li>• Sau khi đăng ký, mã bệnh nhân sẽ được tạo tự động</li>
        </ul>
      </div>
    </div>
  );
}
