// frontend/src/components/patient/MedicalRecordDetail.jsx
import React from 'react';
import { X, Calendar, User, FileText, Activity, Thermometer, Heart, Download } from 'lucide-react';
import { formatDate, formatDateTime } from '../../utils/helpers';
import Modal from '../common/Modal';

export default function MedicalRecordDetail({ isOpen, onClose, record, onDownload }) {
  if (!record) return null;

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b bg-gradient-to-r from-blue-500 to-blue-600">
          <div>
            <h2 className="text-2xl font-bold text-white">Bệnh án</h2>
            <p className="text-blue-100 text-sm mt-1">
              Ngày khám: {formatDate(record.created_at)}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-white hover:text-gray-200 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {/* Doctor Info */}
          <div className="bg-gray-50 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <User className="w-4 h-4 text-blue-600" />
              Bác sĩ khám
            </h3>
            <div className="flex items-center gap-3">
              <img
                src={record.doctor?.user?.avatar_url || '/avatar-placeholder.png'}
                alt={record.doctor?.user?.full_name}
                className="w-12 h-12 rounded-full object-cover"
              />
              <div>
                <p className="font-semibold text-gray-900">
                  BS. {record.doctor?.user?.full_name}
                </p>
                <p className="text-sm text-gray-600">{record.doctor?.specialty}</p>
              </div>
            </div>
          </div>

          {/* Vital Signs */}
          {(record.blood_pressure || record.heart_rate || record.temperature || record.weight || record.height) && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-600" />
                Chỉ số sinh tồn
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {record.blood_pressure && (
                  <div className="bg-red-50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Heart className="w-4 h-4 text-red-600" />
                      <span className="text-xs text-gray-600">Huyết áp</span>
                    </div>
                    <p className="text-lg font-bold text-red-600">
                      {record.blood_pressure} <span className="text-sm">mmHg</span>
                    </p>
                  </div>
                )}
                
                {record.heart_rate && (
                  <div className="bg-pink-50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Activity className="w-4 h-4 text-pink-600" />
                      <span className="text-xs text-gray-600">Nhịp tim</span>
                    </div>
                    <p className="text-lg font-bold text-pink-600">
                      {record.heart_rate} <span className="text-sm">bpm</span>
                    </p>
                  </div>
                )}
                
                {record.temperature && (
                  <div className="bg-orange-50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Thermometer className="w-4 h-4 text-orange-600" />
                      <span className="text-xs text-gray-600">Nhiệt độ</span>
                    </div>
                    <p className="text-lg font-bold text-orange-600">
                      {record.temperature} <span className="text-sm">°C</span>
                    </p>
                  </div>
                )}
                
                {record.weight && (
                  <div className="bg-blue-50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Activity className="w-4 h-4 text-blue-600" />
                      <span className="text-xs text-gray-600">Cân nặng</span>
                    </div>
                    <p className="text-lg font-bold text-blue-600">
                      {record.weight} <span className="text-sm">kg</span>
                    </p>
                  </div>
                )}
                
                {record.height && (
                  <div className="bg-green-50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Activity className="w-4 h-4 text-green-600" />
                      <span className="text-xs text-gray-600">Chiều cao</span>
                    </div>
                    <p className="text-lg font-bold text-green-600">
                      {record.height} <span className="text-sm">cm</span>
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Symptoms */}
          {record.symptoms && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4 text-blue-600" />
                Triệu chứng
              </h3>
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-gray-700 whitespace-pre-line">{record.symptoms}</p>
              </div>
            </div>
          )}

          {/* Diagnosis */}
          {record.diagnosis && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4 text-blue-600" />
                Chẩn đoán
              </h3>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-gray-900 font-medium text-lg">{record.diagnosis}</p>
              </div>
            </div>
          )}

          {/* Treatment Plan */}
          {record.treatment_plan && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4 text-blue-600" />
                Phương pháp điều trị
              </h3>
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <p className="text-gray-700 whitespace-pre-line">{record.treatment_plan}</p>
              </div>
            </div>
          )}

          {/* Notes */}
          {record.notes && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4 text-blue-600" />
                Ghi chú
              </h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-gray-700 whitespace-pre-line">{record.notes}</p>
              </div>
            </div>
          )}

          {/* Follow up */}
          {record.follow_up_date && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <div className="flex items-center gap-2 text-purple-700">
                <Calendar className="w-4 h-4" />
                <span className="text-sm font-medium">
                  Tái khám: {formatDate(record.follow_up_date)}
                </span>
              </div>
              {record.follow_up_notes && (
                <p className="text-sm text-purple-600 mt-2">{record.follow_up_notes}</p>
              )}
            </div>
          )}

          {/* Metadata */}
          <div className="mt-6 pt-6 border-t">
            <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
              <div>
                <span className="font-medium">Mã bệnh án:</span>
                <span className="ml-2">#{record.id}</span>
              </div>
              <div>
                <span className="font-medium">Ngày tạo:</span>
                <span className="ml-2">{formatDateTime(record.created_at)}</span>
              </div>
              {record.updated_at && record.updated_at !== record.created_at && (
                <div className="col-span-2">
                  <span className="font-medium">Cập nhật lần cuối:</span>
                  <span className="ml-2">{formatDateTime(record.updated_at)}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-between items-center p-6 border-t bg-gray-50">
          <div className="text-sm text-gray-600">
            <span className="font-medium">Lưu ý:</span> Đây là tài liệu y tế quan trọng, vui lòng bảo quản cẩn thận
          </div>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Đóng
            </button>
            {onDownload && (
              <button
                onClick={() => onDownload(record)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Tải về PDF
              </button>
            )}
          </div>
        </div>
      </div>
    </Modal>
  );
}