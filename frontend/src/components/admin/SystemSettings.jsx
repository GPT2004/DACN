// frontend/src/components/admin/SystemSettings.jsx
import React, { useState } from 'react';
import { Settings, Save, Database, Bell, Clock, FileText, Shield } from 'lucide-react';
import Button from '../common/Button';
import Input from '../common/Input';

export default function SystemSettings() {
  const [activeTab, setActiveTab] = useState('general');
  const [loading, setLoading] = useState(false);

  // General Settings
  const [generalSettings, setGeneralSettings] = useState({
    clinic_name: 'Phòng khám Đa khoa ABC',
    clinic_address: '123 Đường ABC, Quận 1, TP.HCM',
    clinic_phone: '0123456789',
    clinic_email: 'contact@clinic.com',
    working_hours_start: '07:00',
    working_hours_end: '20:00',
    break_time_start: '12:00',
    break_time_end: '13:00',
  });

  // Appointment Settings
  const [appointmentSettings, setAppointmentSettings] = useState({
    timeslot_duration: 20,
    max_appointments_per_slot: 1,
    booking_advance_days: 30,
    cancel_deadline_hours: 24,
    auto_reminder_hours: 24,
    auto_noshow_hours: 2,
  });

  // Notification Settings
  const [notificationSettings, setNotificationSettings] = useState({
    email_notifications: true,
    sms_notifications: false,
    appointment_reminder: true,
    appointment_confirmation: true,
    low_stock_alert: true,
    expiring_medicine_alert: true,
  });

  // Stock Settings
  const [stockSettings, setStockSettings] = useState({
    low_stock_threshold: 20,
    expiry_warning_days: 30,
    auto_reorder: false,
    reorder_quantity: 100,
  });

  const tabs = [
    { id: 'general', label: 'Tổng quan', icon: Settings },
    { id: 'appointment', label: 'Lịch hẹn', icon: Clock },
    { id: 'notification', label: 'Thông báo', icon: Bell },
    { id: 'stock', label: 'Tồn kho', icon: Database },
    { id: 'backup', label: 'Sao lưu', icon: FileText },
    { id: 'security', label: 'Bảo mật', icon: Shield },
  ];

  const handleSaveSettings = async (settingsType) => {
    try {
      setLoading(true);
      // API call to save settings
      await new Promise(resolve => setTimeout(resolve, 1000));
      alert('Lưu cài đặt thành công!');
    } catch (error) {
      console.error('Save settings error:', error);
      alert('Có lỗi xảy ra!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Settings className="w-8 h-8 text-blue-600" />
        <h2 className="text-2xl font-bold text-gray-800">Cài đặt Hệ thống</h2>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b">
          <div className="flex overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center px-6 py-3 border-b-2 font-medium text-sm whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <Icon className="w-4 h-4 mr-2" />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="p-6">
          {/* General Settings */}
          {activeTab === 'general' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-gray-800">Thông tin Phòng khám</h3>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Tên phòng khám
                  </label>
                  <Input
                    value={generalSettings.clinic_name}
                    onChange={(e) => setGeneralSettings({ ...generalSettings, clinic_name: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Số điện thoại
                  </label>
                  <Input
                    value={generalSettings.clinic_phone}
                    onChange={(e) => setGeneralSettings({ ...generalSettings, clinic_phone: e.target.value })}
                  />
                </div>

                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Địa chỉ
                  </label>
                  <Input
                    value={generalSettings.clinic_address}
                    onChange={(e) => setGeneralSettings({ ...generalSettings, clinic_address: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Email liên hệ
                  </label>
                  <Input
                    type="email"
                    value={generalSettings.clinic_email}
                    onChange={(e) => setGeneralSettings({ ...generalSettings, clinic_email: e.target.value })}
                  />
                </div>
              </div>

              <h3 className="text-lg font-semibold text-gray-800 mt-8">Giờ làm việc</h3>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Giờ bắt đầu
                  </label>
                  <Input
                    type="time"
                    value={generalSettings.working_hours_start}
                    onChange={(e) => setGeneralSettings({ ...generalSettings, working_hours_start: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Giờ kết thúc
                  </label>
                  <Input
                    type="time"
                    value={generalSettings.working_hours_end}
                    onChange={(e) => setGeneralSettings({ ...generalSettings, working_hours_end: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Giờ nghỉ trưa (bắt đầu)
                  </label>
                  <Input
                    type="time"
                    value={generalSettings.break_time_start}
                    onChange={(e) => setGeneralSettings({ ...generalSettings, break_time_start: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Giờ nghỉ trưa (kết thúc)
                  </label>
                  <Input
                    type="time"
                    value={generalSettings.break_time_end}
                    onChange={(e) => setGeneralSettings({ ...generalSettings, break_time_end: e.target.value })}
                  />
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={() => handleSaveSettings('general')} disabled={loading}>
                  <Save className="w-4 h-4 mr-2" />
                  {loading ? 'Đang lưu...' : 'Lưu cài đặt'}
                </Button>
              </div>
            </div>
          )}

          {/* Appointment Settings */}
          {activeTab === 'appointment' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-gray-800">Cài đặt Lịch hẹn</h3>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Thời lượng mỗi slot (phút)
                  </label>
                  <Input
                    type="number"
                    value={appointmentSettings.timeslot_duration}
                    onChange={(e) => setAppointmentSettings({ ...appointmentSettings, timeslot_duration: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Số lịch hẹn tối đa/slot
                  </label>
                  <Input
                    type="number"
                    value={appointmentSettings.max_appointments_per_slot}
                    onChange={(e) => setAppointmentSettings({ ...appointmentSettings, max_appointments_per_slot: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Đặt trước tối đa (ngày)
                  </label>
                  <Input
                    type="number"
                    value={appointmentSettings.booking_advance_days}
                    onChange={(e) => setAppointmentSettings({ ...appointmentSettings, booking_advance_days: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Hạn hủy trước (giờ)
                  </label>
                  <Input
                    type="number"
                    value={appointmentSettings.cancel_deadline_hours}
                    onChange={(e) => setAppointmentSettings({ ...appointmentSettings, cancel_deadline_hours: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Nhắc trước (giờ)
                  </label>
                  <Input
                    type="number"
                    value={appointmentSettings.auto_reminder_hours}
                    onChange={(e) => setAppointmentSettings({ ...appointmentSettings, auto_reminder_hours: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Tự động no-show sau (giờ)
                  </label>
                  <Input
                    type="number"
                    value={appointmentSettings.auto_noshow_hours}
                    onChange={(e) => setAppointmentSettings({ ...appointmentSettings, auto_noshow_hours: e.target.value })}
                  />
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={() => handleSaveSettings('appointment')} disabled={loading}>
                  <Save className="w-4 h-4 mr-2" />
                  {loading ? 'Đang lưu...' : 'Lưu cài đặt'}
                </Button>
              </div>
            </div>
          )}

          {/* Notification Settings */}
          {activeTab === 'notification' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-gray-800">Cài đặt Thông báo</h3>
              
              <div className="space-y-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                    checked={notificationSettings.email_notifications}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, email_notifications: e.target.checked })}
                  />
                  <span className="ml-3">
                    <span className="text-sm font-medium text-gray-700">Thông báo qua Email</span>
                    <p className="text-xs text-gray-500">Gửi thông báo qua email cho người dùng</p>
                  </span>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                    checked={notificationSettings.sms_notifications}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, sms_notifications: e.target.checked })}
                  />
                  <span className="ml-3">
                    <span className="text-sm font-medium text-gray-700">Thông báo qua SMS</span>
                    <p className="text-xs text-gray-500">Gửi SMS cho các thông báo quan trọng</p>
                  </span>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                    checked={notificationSettings.appointment_reminder}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, appointment_reminder: e.target.checked })}
                  />
                  <span className="ml-3">
                    <span className="text-sm font-medium text-gray-700">Nhắc lịch hẹn</span>
                    <p className="text-xs text-gray-500">Tự động nhắc bệnh nhân về lịch hẹn sắp tới</p>
                  </span>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                    checked={notificationSettings.appointment_confirmation}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, appointment_confirmation: e.target.checked })}
                  />
                  <span className="ml-3">
                    <span className="text-sm font-medium text-gray-700">Xác nhận lịch hẹn</span>
                    <p className="text-xs text-gray-500">Gửi thông báo xác nhận khi có lịch hẹn mới</p>
                  </span>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                    checked={notificationSettings.low_stock_alert}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, low_stock_alert: e.target.checked })}
                  />
                  <span className="ml-3">
                    <span className="text-sm font-medium text-gray-700">Cảnh báo tồn kho thấp</span>
                    <p className="text-xs text-gray-500">Thông báo khi thuốc sắp hết</p>
                  </span>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                    checked={notificationSettings.expiring_medicine_alert}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, expiring_medicine_alert: e.target.checked })}
                  />
                  <span className="ml-3">
                    <span className="text-sm font-medium text-gray-700">Cảnh báo thuốc hết hạn</span>
                    <p className="text-xs text-gray-500">Thông báo khi thuốc sắp hết hạn sử dụng</p>
                  </span>
                </label>
              </div>

              <div className="flex justify-end">
                <Button onClick={() => handleSaveSettings('notification')} disabled={loading}>
                  <Save className="w-4 h-4 mr-2" />
                  {loading ? 'Đang lưu...' : 'Lưu cài đặt'}
                </Button>
              </div>
            </div>
          )}

          {/* Stock Settings */}
          {activeTab === 'stock' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-gray-800">Cài đặt Tồn kho</h3>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Ngưỡng cảnh báo tồn kho thấp
                  </label>
                  <Input
                    type="number"
                    value={stockSettings.low_stock_threshold}
                    onChange={(e) => setStockSettings({ ...stockSettings, low_stock_threshold: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Cảnh báo hết hạn trước (ngày)
                  </label>
                  <Input
                    type="number"
                    value={stockSettings.expiry_warning_days}
                    onChange={(e) => setStockSettings({ ...stockSettings, expiry_warning_days: e.target.value })}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Số lượng đặt lại tự động
                  </label>
                  <Input
                    type="number"
                    value={stockSettings.reorder_quantity}
                    onChange={(e) => setStockSettings({ ...stockSettings, reorder_quantity: e.target.value })}
                  />
                </div>
              </div>

              <div className="space-y-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                    checked={stockSettings.auto_reorder}
                    onChange={(e) => setStockSettings({ ...stockSettings, auto_reorder: e.target.checked })}
                  />
                  <span className="ml-3">
                    <span className="text-sm font-medium text-gray-700">Tự động đặt hàng</span>
                    <p className="text-xs text-gray-500">Tự động tạo đơn đặt hàng khi tồn kho thấp</p>
                  </span>
                </label>
              </div>

              <div className="flex justify-end">
                <Button onClick={() => handleSaveSettings('stock')} disabled={loading}>
                  <Save className="w-4 h-4 mr-2" />
                  {loading ? 'Đang lưu...' : 'Lưu cài đặt'}
                </Button>
              </div>
            </div>
          )}

          {/* Backup Settings */}
          {activeTab === 'backup' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-gray-800">Sao lưu Dữ liệu</h3>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                <p className="text-sm text-blue-800 mb-4">
                  Sao lưu dữ liệu định kỳ giúp bảo vệ thông tin quan trọng của phòng khám.
                </p>
                <div className="flex gap-4">
                  <Button>Sao lưu ngay</Button>
                  <Button>Khôi phục dữ liệu</Button>
                </div>
              </div>

              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-sm text-yellow-800">
                  ⚠️ Lần sao lưu gần nhất: 10/11/2025 - 02:00 AM
                </p>
              </div>
            </div>
          )}

          {/* Security Settings */}
          {activeTab === 'security' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-gray-800">Bảo mật</h3>
              
              <div className="space-y-4">
                <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                  <h4 className="font-medium text-red-900 mb-2">Vùng nguy hiểm</h4>
                  <p className="text-sm text-red-700 mb-4">
                    Các hành động sau có thể ảnh hưởng nghiêm trọng đến hệ thống.
                  </p>
                  <div className="flex gap-4">
                    <Button>Đặt lại hệ thống</Button>
                    <Button>Xóa tất cả dữ liệu</Button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}