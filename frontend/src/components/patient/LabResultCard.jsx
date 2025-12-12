// frontend/src/components/patient/LabResultCard.jsx
import React from 'react';
import { FileText, Calendar, User, Download, AlertCircle, CheckCircle, TrendingUp, TrendingDown } from 'lucide-react';
import { formatDate } from '../../utils/helpers';
import Badge from '../common/Badge';

const getResultStatus = (value, normalRange) => {
  if (!normalRange || !value) return 'normal';
  
  const [min, max] = normalRange.split('-').map(v => parseFloat(v.trim()));
  const numValue = parseFloat(value);
  
  if (isNaN(numValue) || isNaN(min) || isNaN(max)) return 'normal';
  
  if (numValue < min) return 'low';
  if (numValue > max) return 'high';
  return 'normal';
};

const getStatusIcon = (status) => {
  switch (status) {
    case 'low':
      return <TrendingDown className="w-4 h-4 text-blue-600" />;
    case 'high':
      return <TrendingUp className="w-4 h-4 text-red-600" />;
    default:
      return <CheckCircle className="w-4 h-4 text-green-600" />;
  }
};

const getStatusColor = (status) => {
  switch (status) {
    case 'low':
      return 'text-blue-600 bg-blue-50';
    case 'high':
      return 'text-red-600 bg-red-50';
    default:
      return 'text-green-600 bg-green-50';
  }
};

const getStatusLabel = (status) => {
  switch (status) {
    case 'low':
      return 'Thấp';
    case 'high':
      return 'Cao';
    default:
      return 'Bình thường';
  }
};

export default function LabResultCard({ 
  labOrder, 
  onDownload,
  onViewDetail 
}) {
  if (!labOrder) return null;

  const hasAbnormalResults = labOrder.results?.some(result => {
    const status = getResultStatus(result.value, result.normal_range);
    return status !== 'normal';
  });

  return (
    <div className="bg-white rounded-lg shadow hover:shadow-md transition-shadow">
      <div className="p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                {labOrder.test_name || 'Xét nghiệm'}
              </h3>
              <div className="flex items-center gap-2 mt-1">
                <Calendar className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600">
                  {formatDate(labOrder.created_at)}
                </span>
              </div>
            </div>
          </div>
          
          {hasAbnormalResults && (
            <div className="flex items-center gap-2 px-3 py-1 bg-yellow-50 text-yellow-800 rounded-lg">
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm font-medium">Có chỉ số bất thường</span>
            </div>
          )}
        </div>

        {/* Doctor Info */}
        {labOrder.doctor && (
          <div className="flex items-center gap-2 mb-4 pb-4 border-b">
            <User className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-600">
              Bác sĩ chỉ định: <span className="font-medium text-gray-900">
                BS. {labOrder.doctor.user?.full_name}
              </span>
            </span>
          </div>
        )}

        {/* Results */}
        {labOrder.results && labOrder.results.length > 0 ? (
          <div className="space-y-3 mb-4">
            <h4 className="text-sm font-semibold text-gray-700">Kết quả xét nghiệm:</h4>
            <div className="space-y-2">
              {labOrder.results.map((result, index) => {
                const status = getResultStatus(result.value, result.normal_range);
                return (
                  <div key={index} className="bg-gray-50 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-gray-700">
                        {result.test_name}
                      </span>
                      <div className="flex items-center gap-2">
                        {getStatusIcon(status)}
                        <span className={`text-sm font-semibold px-2 py-0.5 rounded ${getStatusColor(status)}`}>
                          {result.value} {result.unit}
                        </span>
                      </div>
                    </div>
                    {result.normal_range && (
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>Chỉ số bình thường: {result.normal_range} {result.unit}</span>
                        {status !== 'normal' && (
                          <span className={`font-medium ${getStatusColor(status)}`}>
                            {getStatusLabel(status)}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <div className="flex items-center gap-2 text-yellow-800">
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm font-medium">
                Kết quả xét nghiệm đang được xử lý
              </span>
            </div>
          </div>
        )}

        {/* Notes */}
        {labOrder.notes && (
          <div className="bg-blue-50 rounded-lg p-3 mb-4">
            <p className="text-sm text-blue-700">
              <span className="font-medium">Ghi chú:</span> {labOrder.notes}
            </p>
          </div>
        )}

        {/* Status */}
        <div className="mb-4">
          <Badge 
            text={labOrder.status === 'COMPLETED' ? 'Đã hoàn thành' : 
                  labOrder.status === 'IN_PROGRESS' ? 'Đang xử lý' : 'Chờ xử lý'} 
          />
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-4 border-t">
          {onViewDetail && (
            <button
              onClick={() => onViewDetail(labOrder)}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
            >
              Xem chi tiết
            </button>
          )}
          {onDownload && labOrder.status === 'COMPLETED' && (
            <button
              onClick={() => onDownload(labOrder)}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Tải về
            </button>
          )}
        </div>

        {/* Warning for abnormal results */}
        {hasAbnormalResults && (
          <div className="mt-4 bg-orange-50 border border-orange-200 rounded-lg p-3">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-orange-600 mt-0.5" />
              <p className="text-sm text-orange-700">
                <span className="font-medium">Lưu ý:</span> Có một số chỉ số bất thường. 
                Vui lòng tham khảo ý kiến bác sĩ để được tư vấn chi tiết.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}