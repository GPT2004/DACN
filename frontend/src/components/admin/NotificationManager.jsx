/* eslint-disable no-console */
// frontend/src/components/admin/NotificationManager.jsx
import React, { useState } from 'react';
import { Send, Bell, Users } from 'lucide-react';
import { notificationService } from '../../services/notificationService';
import Button from '../common/Button';
import Input from '../common/Input';

export default function NotificationManager() {
  const [activeTab, setActiveTab] = useState('single');
  const [loading, setLoading] = useState(false);
  
  // Single notification
  const [singleForm, setSingleForm] = useState({
    user_id: '',
    type: 'INFO',
    title: '',
    message: '',
  });

  // Broadcast notification
  const [broadcastForm, setBroadcastForm] = useState({
    role_name: '',
    type: 'INFO',
    title: '',
    message: '',
  });

  const notificationTypes = [
    { value: 'INFO', label: 'Thông tin' },
    { value: 'SUCCESS', label: 'Thành công' },
    { value: 'WARNING', label: 'Cảnh báo' },
    { value: 'ERROR', label: 'Lỗi' },
  ];

  const roles = [
    { value: 'Admin', label: 'Admin' },
    { value: 'Doctor', label: 'Bác sĩ' },
    { value: 'Patient', label: 'Bệnh nhân' },
    { value: 'Receptionist', label: 'Lễ tân' },
    { value: 'Pharmacist', label: 'Dược sĩ' },
    { value: 'LabTech', label: 'Kỹ thuật viên' },
  ];

  const handleSendSingle = async (e) => {
    e.preventDefault();
    
    if (!singleForm.user_id || !singleForm.title || !singleForm.message) {
      alert('Vui lòng điền đầy đủ thông tin!');
      return;
    }

    try {
      setLoading(true);
      await notificationService.sendNotification(
        singleForm.user_id,
        singleForm.type,
        {
          title: singleForm.title,
          message: singleForm.message,
        }
      );
      
      alert('Gửi thông báo thành công!');
      setSingleForm({
        user_id: '',
        type: 'INFO',
        title: '',
        message: '',
      });
    } catch (error) {
      console.error('Send notification error:', error);
      alert('Có lỗi xảy ra!');
    } finally {
      setLoading(false);
    }
  };

  const handleBroadcast = async (e) => {
    e.preventDefault();
    
    if (!broadcastForm.role_name || !broadcastForm.title || !broadcastForm.message) {
      alert('Vui lòng điền đầy đủ thông tin!');
      return;
    }

    try {
      setLoading(true);
      await notificationService.broadcastNotification(
        broadcastForm.role_name,
        broadcastForm.type,
        {
          title: broadcastForm.title,
          message: broadcastForm.message,
        }
      );
      
      alert('Gửi thông báo broadcast thành công!');
      setBroadcastForm({
        role_name: '',
        type: 'INFO',
        title: '',
        message: '',
      });
    } catch (error) {
      console.error('Broadcast notification error:', error);
      alert('Có lỗi xảy ra!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Bell className="w-8 h-8 text-blue-600" />
        <h2 className="text-2xl font-bold text-gray-800">Quản lý Thông báo</h2>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b">
          <div className="flex">
            <button
              onClick={() => setActiveTab('single')}
              className={`flex items-center px-6 py-3 border-b-2 font-medium text-sm ${
                activeTab === 'single'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <Send className="w-4 h-4 mr-2" />
              Gửi cá nhân
            </button>
            <button
              onClick={() => setActiveTab('broadcast')}
              className={`flex items-center px-6 py-3 border-b-2 font-medium text-sm ${
                activeTab === 'broadcast'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <Users className="w-4 h-4 mr-2" />
              Gửi theo vai trò
            </button>
          </div>
        </div>

        <div className="p-6">
          {/* Single Notification Form */}
          {activeTab === 'single' && (
            <form onSubmit={handleSendSingle} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  ID Người dùng <span className="text-red-500">*</span>
                </label>
                <Input
                  type="number"
                  value={singleForm.user_id}
                  onChange={(e) => setSingleForm({ ...singleForm, user_id: e.target.value })}
                  placeholder="Nhập ID người dùng"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Tìm ID người dùng trong danh sách quản lý người dùng
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Loại thông báo <span className="text-red-500">*</span>
                </label>
                <select
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  value={singleForm.type}
                  onChange={(e) => setSingleForm({ ...singleForm, type: e.target.value })}
                >
                  {notificationTypes.map((type) => (
                    <option key={type.value} value={type.value}>
                      {type.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tiêu đề <span className="text-red-500">*</span>
                </label>
                <Input
                  value={singleForm.title}
                  onChange={(e) => setSingleForm({ ...singleForm, title: e.target.value })}
                  placeholder="Nhập tiêu đề thông báo"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Nội dung <span className="text-red-500">*</span>
                </label>
                <textarea
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  rows="4"
                  value={singleForm.message}
                  onChange={(e) => setSingleForm({ ...singleForm, message: e.target.value })}
                  placeholder="Nhập nội dung thông báo..."
                />
              </div>

              <div className="flex justify-end">
                <Button type="submit" disabled={loading}>
                  <Send className="w-4 h-4 mr-2" />
                  {loading ? 'Đang gửi...' : 'Gửi thông báo'}
                </Button>
              </div>
            </form>
          )}

          {/* Broadcast Notification Form */}
          {activeTab === 'broadcast' && (
            <form onSubmit={handleBroadcast} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Vai trò <span className="text-red-500">*</span>
                </label>
                <select
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  value={broadcastForm.role_name}
                  onChange={(e) => setBroadcastForm({ ...broadcastForm, role_name: e.target.value })}
                >
                  <option value="">Chọn vai trò</option>
                  {roles.map((role) => (
                    <option key={role.value} value={role.value}>
                      {role.label}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  Thông báo sẽ được gửi đến tất cả người dùng có vai trò này
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Loại thông báo <span className="text-red-500">*</span>
                </label>
                <select
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  value={broadcastForm.type}
                  onChange={(e) => setBroadcastForm({ ...broadcastForm, type: e.target.value })}
                >
                  {notificationTypes.map((type) => (
                    <option key={type.value} value={type.value}>
                      {type.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tiêu đề <span className="text-red-500">*</span>
                </label>
                <Input
                  value={broadcastForm.title}
                  onChange={(e) => setBroadcastForm({ ...broadcastForm, title: e.target.value })}
                  placeholder="Nhập tiêu đề thông báo"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Nội dung <span className="text-red-500">*</span>
                </label>
                <textarea
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  rows="4"
                  value={broadcastForm.message}
                  onChange={(e) => setBroadcastForm({ ...broadcastForm, message: e.target.value })}
                  placeholder="Nhập nội dung thông báo..."
                />
              </div>

              <div className="flex justify-end">
                <Button type="submit" disabled={loading}>
                  <Users className="w-4 h-4 mr-2" />
                  {loading ? 'Đang gửi...' : 'Gửi broadcast'}
                </Button>
              </div>
            </form>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">Mẫu thông báo nhanh</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white rounded-lg p-4 border border-blue-100">
            <h4 className="font-medium text-gray-900 mb-2">Nhắc lịch hẹn</h4>
            <p className="text-sm text-gray-600 mb-3">
              Nhắc nhở bệnh nhân về lịch hẹn sắp tới
            </p>
            <Button
              onClick={() => {
                if (activeTab === 'broadcast') {
                  setBroadcastForm({
                    ...broadcastForm,
                    type: 'INFO',
                    title: 'Nhắc lịch hẹn',
                    message: 'Bạn có lịch hẹn khám bệnh vào ngày mai. Vui lòng đến đúng giờ.',
                  });
                }
              }}
            >
              Sử dụng
            </Button>
          </div>

          <div className="bg-white rounded-lg p-4 border border-blue-100">
            <h4 className="font-medium text-gray-900 mb-2">Thông báo bảo trì</h4>
            <p className="text-sm text-gray-600 mb-3">
              Thông báo hệ thống bảo trì
            </p>
            <Button
              onClick={() => {
                if (activeTab === 'broadcast') {
                  setBroadcastForm({
                    ...broadcastForm,
                    type: 'WARNING',
                    title: 'Thông báo bảo trì hệ thống',
                    message: 'Hệ thống sẽ bảo trì từ 23:00 - 02:00 sáng. Vui lòng hoàn tất công việc trước thời gian này.',
                  });
                }
              }}
            >
              Sử dụng
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}