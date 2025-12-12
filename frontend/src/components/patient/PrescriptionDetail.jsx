// frontend/src/components/patient/PrescriptionDetail.jsx
import React from 'react';
import { X, Calendar, User, Pill, FileText, Download, CheckCircle, AlertCircle, Clock, Info } from 'lucide-react';
import { formatDate, formatDateTime, formatCurrency } from '../../utils/helpers';
import Modal from '../common/Modal';

const getStatusInfo = (status) => {
  const statusMap = {
    DRAFT: { label: 'Bản nháp', color: 'bg-gray-100 text-gray-800', icon: Clock },
    APPROVED: { label: 'Đã duyệt', color: 'bg-blue-100 text-blue-800', icon: CheckCircle },
    DISPENSED: { label: 'Đã phát thuốc', color: 'bg-green-100 text-green-800', icon: CheckCircle },
  };
  return statusMap[status] || statusMap.DRAFT;
};

export default function PrescriptionDetail({ isOpen, onClose, prescription, onDownload }) {
  if (!prescription) return null;

  const statusInfo = getStatusInfo(prescription.status);
  const StatusIcon = statusInfo.icon;

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b bg-gradient-to-r from-green-500 to-green-600">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <Pill className="w-6 h-6" />
              Đơn thuốc #{prescription.id}
            </h2>
            <p className="text-green-100 text-sm mt-1">
              Ngày kê: {formatDate(prescription.created_at)}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1 bg-white/20 rounded-lg">
              <StatusIcon className="w-4 h-4 text-white" />
              <span className="text-white text-sm font-medium">{statusInfo.label}</span>
            </div>
            <button
              onClick={onClose}
              className="text-white hover:text-gray-200 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {/* Doctor Info */}
          <div className="bg-gray-50 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <User className="w-4 h-4 text-green-600" />
              Bác sĩ kê đơn
            </h3>
            <div className="flex items-center gap-3">
              <img
                src={prescription.doctor?.user?.avatar_url || '/avatar-placeholder.png'}
                alt={prescription.doctor?.user?.full_name}
                className="w-12 h-12 rounded-full object-cover"
              />
              <div>
                <p className="font-semibold text-gray-900">
                  BS. {prescription.doctor?.user?.full_name}
                </p>
                <p className="text-sm text-gray-600">{prescription.doctor?.specialty}</p>
                {prescription.doctor?.license_number && (
                  <p className="text-xs text-gray-500 mt-1">
                    CCHN: {prescription.doctor.license_number}
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Diagnosis */}
          {prescription.diagnosis && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4 text-green-600" />
                Chẩn đoán
              </h3>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-gray-900">{prescription.diagnosis}</p>
              </div>
            </div>
          )}

          {/* Medicines */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <Pill className="w-4 h-4 text-green-600" />
              Danh sách thuốc ({prescription.items?.length || 0} loại)
            </h3>
            <div className="space-y-3">
              {prescription.items?.map((item, index) => (
                <div key={index} className="bg-white border-2 border-gray-200 rounded-lg p-4 hover:border-green-300 transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="w-6 h-6 bg-green-100 text-green-700 rounded-full flex items-center justify-center text-xs font-semibold">
                          {index + 1}
                        </span>
                        <h4 className="font-semibold text-gray-900 text-lg">{item.medicine?.name}</h4>
                      </div>
                      
                      {item.medicine?.active_ingredient && (
                        <p className="text-sm text-gray-600 mb-2">
                          <span className="font-medium">Hoạt chất:</span> {item.medicine.active_ingredient}
                        </p>
                      )}
                    </div>
                    {item.price && (
                      <div className="text-right ml-4">
                        <p className="text-xs text-gray-500 mb-1">Thành tiền</p>
                        <p className="text-lg font-bold text-green-600">
                          {formatCurrency(item.price * item.quantity)}
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div className="bg-gray-50 rounded-lg p-3">
                      <p className="text-xs text-gray-600 mb-1">Liều dùng</p>
                      <p className="font-semibold text-gray-900">{item.dosage}</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <p className="text-xs text-gray-600 mb-1">Số lượng</p>
                      <p className="font-semibold text-gray-900">
                        {item.quantity} {item.unit || 'viên'}
                      </p>
                    </div>
                  </div>

                  {item.instructions && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                      <div className="flex items-start gap-2">
                        <Info className="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="text-xs font-medium text-yellow-800 mb-1">Cách dùng:</p>
                          <p className="text-sm text-yellow-700">{item.instructions}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Total Amount */}
          {prescription.total_amount && (
            <div className="bg-gradient-to-r from-green-50 to-blue-50 border-2 border-green-200 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <span className="text-lg font-semibold text-gray-700">Tổng tiền thuốc:</span>
                <span className="text-2xl font-bold text-green-600">
                  {formatCurrency(prescription.total_amount)}
                </span>
              </div>
            </div>
          )}

          {/* General Notes */}
          {prescription.notes && (
            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4 text-green-600" />
                Lưu ý của bác sĩ
              </h3>
              <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                <p className="text-gray-700 whitespace-pre-line">{prescription.notes}</p>
              </div>
            </div>
          )}

          {/* Important Instructions */}
          <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="font-semibold text-red-800 mb-2">Lưu ý quan trọng khi dùng thuốc:</h4>
                <ul className="text-sm text-red-700 space-y-2 list-disc list-inside">
                  <li>Uống đúng liều lượng và thời gian theo chỉ định của bác sĩ</li>
                  <li>Không tự ý tăng, giảm liều hoặc ngừng thuốc</li>
                  <li>Bảo quản thuốc nơi khô ráo, tránh ánh sáng trực tiếp và nhiệt độ cao</li>
                  <li>Để xa tầm tay trẻ em</li>
                  <li>Kiểm tra hạn sử dụng trước khi dùng</li>
                  <li>Nếu xuất hiện các triệu chứng bất thường, hãy liên hệ bác sĩ ngay</li>
                  <li>Không dùng khi đã hết hạn sử dụng</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Dispensed Info */}
          {prescription.status === 'DISPENSED' && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
              <div className="flex items-center gap-2 text-green-700 mb-2">
                <CheckCircle className="w-5 h-5" />
                <span className="font-semibold">Đã phát thuốc</span>
              </div>
              <div className="text-sm text-green-600 space-y-1 ml-7">
                <p>Ngày phát: {formatDateTime(prescription.dispensed_at)}</p>
                {prescription.dispensed_by && (
                  <p>Dược sĩ: {prescription.dispensed_by}</p>
                )}
              </div>
            </div>
          )}

          {/* Metadata */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Thông tin đơn thuốc</h4>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-600">Mã đơn thuốc:</span>
                <span className="ml-2 font-medium text-gray-900">#{prescription.id}</span>
              </div>
              <div>
                <span className="text-gray-600">Ngày kê đơn:</span>
                <span className="ml-2 font-medium text-gray-900">{formatDateTime(prescription.created_at)}</span>
              </div>
              {prescription.approved_at && (
                <div>
                  <span className="text-gray-600">Ngày duyệt:</span>
                  <span className="ml-2 font-medium text-gray-900">{formatDateTime(prescription.approved_at)}</span>
                </div>
              )}
              {prescription.updated_at && prescription.updated_at !== prescription.created_at && (
                <div>
                  <span className="text-gray-600">Cập nhật cuối:</span>
                  <span className="ml-2 font-medium text-gray-900">{formatDateTime(prescription.updated_at)}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-between items-center p-6 border-t bg-gray-50">
          <div className="text-sm text-gray-600">
            <p className="font-medium mb-1">Đơn thuốc này chỉ có hiệu lực trong 30 ngày kể từ ngày kê</p>
            <p className="text-xs">Bảo quản đơn thuốc để tham khảo khi cần thiết</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Đóng
            </button>
            {onDownload && prescription.status === 'APPROVED' && (
              <button
                onClick={() => onDownload(prescription)}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
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