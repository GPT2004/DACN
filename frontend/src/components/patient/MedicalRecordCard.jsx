// frontend/src/components/patient/MedicalRecordCard.jsx
import React from 'react';
import { FileText, User, Calendar, Eye, Download, Pill } from 'lucide-react';
import { formatDate, formatDateTime } from '../../utils/helpers';
import Button from '../common/Button';
import Badge from '../common/Badge';

export default function MedicalRecordCard({ record, onView }) {
  const hasPrescription = record.prescription_id || record.has_prescription;
  const hasLabOrder = record.lab_order_id || record.has_lab_order;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-full bg-green-100 text-green-600 flex items-center justify-center">
            <FileText className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">
              Bệnh án #{record.id}
            </h3>
            <div className="flex items-center gap-2 text-sm text-gray-600 mt-1">
              <Calendar className="w-4 h-4" />
              {formatDateTime(record.created_at)}
            </div>
          </div>
        </div>
      </div>

      {/* Doctor Info */}
      <div className="mb-4 pb-4 border-b border-gray-100">
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <User className="w-4 h-4 text-gray-400" />
          <span className="font-medium">Bác sĩ:</span>
          <span>BS. {record.doctor?.user?.full_name || 'N/A'}</span>
        </div>
        {record.doctor?.specialty && (
          <div className="mt-2">
            <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs">
              {record.doctor.specialty}
            </span>
          </div>
        )}
      </div>

      {/* Diagnosis */}
      {record.diagnosis && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-2">Chẩn đoán</h4>
          <p className="text-sm text-gray-700 bg-gray-50 p-3 rounded-lg">
            {record.diagnosis}
          </p>
        </div>
      )}

      {/* Chief Complaint */}
      {record.chief_complaint && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-2">Lý do khám</h4>
          <p className="text-sm text-gray-600">
            {record.chief_complaint}
          </p>
        </div>
      )}

      {/* Treatment */}
      {record.treatment && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-2">Điều trị</h4>
          <p className="text-sm text-gray-600 line-clamp-2">
            {record.treatment}
          </p>
        </div>
      )}

      {/* Related Items */}
      <div className="flex gap-2 mb-4">
        {hasPrescription && (
          <span className="flex items-center gap-1 px-2 py-1 bg-purple-50 text-purple-700 rounded text-xs">
            <Pill className="w-3 h-3" />
            Có đơn thuốc
          </span>
        )}
        {hasLabOrder && (
          <span className="flex items-center gap-1 px-2 py-1 bg-cyan-50 text-cyan-700 rounded text-xs">
            <FileText className="w-3 h-3" />
            Có chỉ định XN
          </span>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <Button
          onClick={() => onView(record)}
          className="flex-1"
          variant="outline"
        >
          <Eye className="w-4 h-4 mr-2" />
          Xem chi tiết
        </Button>
      </div>

      {/* Footer Info */}
      {record.notes && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <p className="text-xs text-gray-500">
            <span className="font-medium">Ghi chú:</span> {record.notes}
          </p>
        </div>
      )}
    </div>
  );
}