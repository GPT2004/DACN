/* eslint-disable no-unused-vars */
// frontend/src/components/patient/PrescriptionCard.jsx
import React from 'react';
import { Pill, Calendar, User, FileText, Download, CheckCircle, Clock, AlertCircle } from 'lucide-react';
import { formatDate, formatCurrency } from '../../utils/helpers';
import Badge from '../common/Badge';

const getStatusInfo = (status) => {
  const statusMap = {
    DRAFT: {
      label: 'Bản nháp',
      color: 'bg-gray-100 text-gray-800',
      icon: Clock,
    },
    APPROVED: {
      label: 'Đã duyệt',
      color: 'bg-blue-100 text-blue-800',
      icon: CheckCircle,
    },
    DISPENSED: {
      label: 'Đã phát thuốc',
      color: 'bg-green-100 text-green-800',
      icon: CheckCircle,
    },
  };
  return statusMap[status] || statusMap.DRAFT;
};

export default function PrescriptionCard({ 
  prescription, 
  onViewDetail,
  onDownload 
}) {
  if (!prescription) return null;

  const statusInfo = getStatusInfo(prescription.status);
  const StatusIcon = statusInfo.icon;

  // Calculate total medicines count
  const totalMedicines = prescription.items?.length || 0;

  return (
    <div className="bg-white rounded-lg shadow hover:shadow-md transition-shadow">
      <div className="p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
              <Pill className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                Đơn thuốc #{prescription.id}
              </h3>
              <div className="flex items-center gap-2 mt-1">
                <Calendar className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600">
                  {formatDate(prescription.created_at)}
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <StatusIcon className="w-4 h-4" />
            <span className={`px-3 py-1 text-xs font-medium rounded-full ${statusInfo.color}`}>
              {statusInfo.label}
            </span>
          </div>
        </div>

        {/* Doctor Info */}
        {prescription.doctor && (
          <div className="flex items-center gap-2 mb-4 pb-4 border-b">
            <User className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-600">
              Bác sĩ kê đơn: <span className="font-medium text-gray-900">
                BS. {prescription.doctor.user?.full_name}
              </span>
            </span>
          </div>
        )}

        {/* Medicines Summary */}
        {prescription.items && prescription.items.length > 0 && (
          <div className="space-y-2 mb-4">
            <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
              <Pill className="w-4 h-4 text-gray-500" />
              Thuốc được kê ({totalMedicines} loại)
            </h4>
            <div className="space-y-2">
              {prescription.items.slice(0, 3).map((item, index) => (
                <div key={index} className="bg-gray-50 rounded-lg p-3">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="font-medium text-gray-900">{item.medicine?.name}</p>
                      <p className="text-sm text-gray-600 mt-1">
                        {item.dosage} - {item.quantity} {item.unit || 'viên'}
                      </p>
                      {item.instructions && (
                        <p className="text-xs text-gray-500 mt-1 italic">
                          {item.instructions}
                        </p>
                      )}
                    </div>
                    {item.price && (
                      <div className="text-right ml-4">
                        <p className="text-sm font-semibold text-gray-900">
                          {formatCurrency(item.price * item.quantity)}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {prescription.items.length > 3 && (
                <div className="text-center py-2">
                  <button
                    onClick={() => onViewDetail && onViewDetail(prescription)}
                    className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                  >
                    Xem thêm {prescription.items.length - 3} thuốc khác
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Total Amount */}
        {prescription.total_amount && (
          <div className="bg-blue-50 rounded-lg p-3 mb-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-blue-700">Tổng tiền:</span>
              <span className="text-lg font-bold text-blue-700">
                {formatCurrency(prescription.total_amount)}
              </span>
            </div>
          </div>
        )}

        {/* Notes */}
        {prescription.notes && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
            <div className="flex items-start gap-2">
              <FileText className="w-4 h-4 text-yellow-600 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-yellow-800">Lưu ý:</p>
                <p className="text-sm text-yellow-700 mt-1">{prescription.notes}</p>
              </div>
            </div>
          </div>
        )}

        {/* Dispensed Info */}
        {prescription.status === 'DISPENSED' && prescription.dispensed_at && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4">
            <div className="flex items-center gap-2 text-green-700">
              <CheckCircle className="w-4 h-4" />
              <span className="text-sm font-medium">
                Đã phát thuốc: {formatDate(prescription.dispensed_at)}
              </span>
            </div>
            {prescription.dispensed_by && (
              <p className="text-xs text-green-600 mt-1 ml-6">
                Dược sĩ: {prescription.dispensed_by}
              </p>
            )}
          </div>
        )}

        {/* Usage Instructions */}
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 mb-4">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-orange-600 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-orange-800">Hướng dẫn sử dụng:</p>
              <ul className="text-sm text-orange-700 mt-1 space-y-1 list-disc list-inside">
                <li>Uống đúng liều lượng và thời gian theo chỉ định</li>
                <li>Không tự ý thay đổi liều lượng</li>
                <li>Bảo quản thuốc nơi khô ráo, tránh ánh sáng</li>
                <li>Liên hệ bác sĩ nếu có tác dụng phụ</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-4 border-t">
          {onViewDetail && (
            <button
              onClick={() => onViewDetail(prescription)}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
            >
              Xem chi tiết
            </button>
          )}
          {onDownload && prescription.status === 'APPROVED' && (
            <button
              onClick={() => onDownload(prescription)}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Tải PDF
            </button>
          )}
        </div>

        {/* Metadata */}
        <div className="mt-4 pt-4 border-t">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>Mã đơn: #{prescription.id}</span>
            {prescription.updated_at && prescription.updated_at !== prescription.created_at && (
              <span>Cập nhật: {formatDate(prescription.updated_at)}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}