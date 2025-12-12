// frontend/src/components/admin/DoctorManagement/DoctorDetail.jsx
import React, { useState, useEffect } from 'react';
import { X, Mail, Phone, Calendar, Award, Briefcase, DollarSign } from 'lucide-react';
import { doctorService } from '../../../services/doctorService';
import { formatDate, formatCurrency } from '../../../utils/helpers';
import Modal from '../../common/Modal';
import Loader from '../../common/Loader';
import Badge from '../../common/Badge';

export default function DoctorDetail({ isOpen, onClose, doctorId }) {
  const [doctor, setDoctor] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (doctorId) {
      fetchDoctorDetail();
      fetchDoctorStats();
    }
  }, [doctorId]);

  const fetchDoctorDetail = async () => {
    try {
      const response = await doctorService.getDoctorById(doctorId);
      setDoctor(response.data);
    } catch (error) {
      console.error('Fetch doctor detail error:', error);
    }
  };

  const fetchDoctorStats = async () => {
    try {
      const response = await doctorService.getDoctorStats(doctorId);
      setStats(response.data);
    } catch (error) {
      console.error('Fetch doctor stats error:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Modal isOpen={isOpen} onClose={onClose}>
        <div className="bg-white rounded-lg p-12">
          <Loader />
        </div>
      </Modal>
    );
  }

  if (!doctor) return null;

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b bg-gradient-to-r from-blue-500 to-blue-600">
          <h2 className="text-2xl font-bold text-white">Thông tin Bác sĩ</h2>
          <button
            onClick={onClose}
            className="text-white hover:text-gray-200"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          {/* Profile Section */}
          <div className="flex items-start gap-6 mb-8">
            <img
              src={doctor.user?.avatar_url || '/avatar-placeholder.png'}
              alt={doctor.user?.full_name}
              className="w-32 h-32 rounded-full object-cover border-4 border-blue-100"
            />
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">
                BS. {doctor.user?.full_name}
              </h3>
              <Badge text={doctor.specialty} />
              <div className="mt-4 space-y-2">
                <div className="flex items-center text-gray-600">
                  <Mail className="w-5 h-5 mr-3" />
                  {doctor.user?.email}
                </div>
                <div className="flex items-center text-gray-600">
                  <Phone className="w-5 h-5 mr-3" />
                  {doctor.user?.phone}
                </div>
                <div className="flex items-center text-gray-600">
                  <Calendar className="w-5 h-5 mr-3" />
                  Ngày sinh: {formatDate(doctor.user?.dob)}
                </div>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-4 gap-4 mb-8">
            <div className="bg-blue-50 rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-blue-600">
                {stats?.totalAppointments || 0}
              </div>
              <div className="text-sm text-gray-600 mt-1">Lịch hẹn</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-green-600">
                {stats?.completedAppointments || 0}
              </div>
              <div className="text-sm text-gray-600 mt-1">Hoàn thành</div>
            </div>
            <div className="bg-yellow-50 rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-yellow-600">
                {stats?.totalPatients || 0}
              </div>
              <div className="text-sm text-gray-600 mt-1">Bệnh nhân</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-purple-600">
                {doctor.years_of_experience || 0}
              </div>
              <div className="text-sm text-gray-600 mt-1">Năm KN</div>
            </div>
          </div>

          {/* Professional Info */}
          <div className="space-y-6">
            <div>
              <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                <Award className="w-5 h-5 mr-2 text-blue-600" />
                Thông tin chuyên môn
              </h4>
              <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Chứng chỉ hành nghề:</span>
                  <span className="font-medium">{doctor.license_number}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Số năm kinh nghiệm:</span>
                  <span className="font-medium">{doctor.years_of_experience} năm</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Phí khám:</span>
                  <span className="font-medium text-green-600">
                    {formatCurrency(doctor.consultation_fee)}
                  </span>
                </div>
              </div>
            </div>

            {doctor.education && (
              <div>
                <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <Briefcase className="w-5 h-5 mr-2 text-blue-600" />
                  Học vấn & Đào tạo
                </h4>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-gray-700 whitespace-pre-line">{doctor.education}</p>
                </div>
              </div>
            )}

            {doctor.bio && (
              <div>
                <h4 className="text-lg font-semibold text-gray-800 mb-3">
                  Giới thiệu
                </h4>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-gray-700 whitespace-pre-line">{doctor.bio}</p>
                </div>
              </div>
            )}

            {/* Status */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Trạng thái tài khoản:</span>
                <Badge 
                  text={doctor.user?.is_active ? 'Hoạt động' : 'Ngừng hoạt động'}
                />
              </div>
              <div className="flex justify-between items-center mt-2">
                <span className="text-gray-600">Ngày tạo:</span>
                <span className="font-medium">{formatDate(doctor.created_at)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Modal>
  );
}