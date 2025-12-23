/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
// frontend/src/components/admin/PatientManagement/PatientDetail.jsx
import React, { useState, useEffect } from 'react';
import { X, User, Calendar, Phone, Mail, MapPin, Heart, AlertCircle, FileText, Pill, TestTube } from 'lucide-react';
import { patientService } from '../../../services/patientService';
import { formatDate, calculateAge } from '../../../utils/helpers';
import Modal from '../../common/Modal';
import Loader from '../../common/Loader';
import Badge from '../../common/Badge';

export default function PatientDetail({ isOpen, onClose, patientId }) {
  const [patient, setPatient] = useState(null);
  const [appointments, setAppointments] = useState([]);
  const [medicalRecords, setMedicalRecords] = useState([]);
  const [prescriptions, setPrescriptions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('info');

  useEffect(() => {
    if (patientId) {
      fetchPatientDetail();
      fetchPatientHistory();
    }
  }, [patientId]);

  const fetchPatientDetail = async () => {
    try {
      const response = await patientService.getPatientById(patientId);
      setPatient(response.data);
    } catch (error) {
      console.error('Fetch patient detail error:', error);
    }
  };

  const fetchPatientHistory = async () => {
    try {
      const [apptRes, recordsRes, prescRes] = await Promise.all([
        patientService.getPatientAppointments(patientId, { limit: 5 }),
        patientService.getPatientMedicalRecords(patientId, { limit: 5 }),
        patientService.getPatientPrescriptions(patientId, { limit: 5 })
      ]);

      setAppointments(apptRes.data.appointments || []);
      setMedicalRecords(recordsRes.data.medicalRecords || []);
      setPrescriptions(prescRes.data.prescriptions || []);
    } catch (error) {
      console.error('Fetch patient history error:', error);
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

  if (!patient) return null;

  const tabs = [
    { id: 'info', label: 'Thông tin', icon: User },
    { id: 'appointments', label: 'Lịch hẹn', icon: Calendar },
    { id: 'records', label: 'Bệnh án', icon: FileText },
    { id: 'prescriptions', label: 'Đơn thuốc', icon: Pill },
  ];

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b bg-gradient-to-r from-green-500 to-green-600">
          <h2 className="text-2xl font-bold text-white">Hồ sơ Bệnh nhân</h2>
          <button
            onClick={onClose}
            className="text-white hover:text-gray-200"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Patient Profile */}
        <div className="p-6 border-b bg-gray-50">
          <div className="flex items-start gap-6">
            <img
              src={patient.user?.avatar_url || '/avatar-placeholder.png'}
              alt={patient.user?.full_name}
              className="w-24 h-24 rounded-full object-cover border-4 border-white shadow-lg"
            />
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-gray-800 mb-2">
                {patient.user?.full_name}
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="flex items-center text-gray-600">
                  <User className="w-4 h-4 mr-2" />
                  {patient.gender === 'Male' ? 'Nam' : patient.gender === 'Female' ? 'Nữ' : 'Khác'} - {calculateAge(patient.user?.dob)} tuổi
                </div>
                <div className="flex items-center text-gray-600">
                  <Phone className="w-4 h-4 mr-2" />
                  {patient.user?.phone}
                </div>
                <div className="flex items-center text-gray-600">
                  <Mail className="w-4 h-4 mr-2" />
                  {patient.user?.email}
                </div>
                <div className="flex items-center text-gray-600">
                  <Calendar className="w-4 h-4 mr-2" />
                  {formatDate(patient.user?.dob)}
                </div>
              </div>
            </div>
            <div className="text-right">
              <Badge text={patient.user?.is_active ? 'Hoạt động' : 'Ngừng hoạt động'} />
              <div className="text-sm text-gray-500 mt-2">
                Mã BN: {patient.id}
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b">
          <div className="flex">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center px-6 py-3 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-green-500 text-green-600'
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

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-350px)]">
          {activeTab === 'info' && (
            <div className="space-y-6">
              {/* Health Info */}
              <div>
                <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                  <Heart className="w-5 h-5 mr-2 text-red-500" />
                  Thông tin sức khỏe
                </h4>
                <div className="bg-gray-50 rounded-lg p-4 grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-gray-600">Nhóm máu:</span>
                    <span className="font-medium ml-2">{patient.blood_type || 'Chưa xác định'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Chiều cao:</span>
                    <span className="font-medium ml-2">{patient.height ? `${patient.height} cm` : 'Chưa có'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Cân nặng:</span>
                    <span className="font-medium ml-2">{patient.weight ? `${patient.weight} kg` : 'Chưa có'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">BMI:</span>
                    <span className="font-medium ml-2">
                      {patient.height && patient.weight 
                        ? (patient.weight / ((patient.height / 100) ** 2)).toFixed(1)
                        : 'Chưa có'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Allergies */}
              {patient.allergies && (
                <div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                    <AlertCircle className="w-5 h-5 mr-2 text-yellow-500" />
                    Dị ứng
                  </h4>
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <p className="text-gray-700">{patient.allergies}</p>
                  </div>
                </div>
              )}

              {/* Medical History */}
              {patient.medical_history && (
                <div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-3">
                    Tiền sử bệnh
                  </h4>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-gray-700 whitespace-pre-line">{patient.medical_history}</p>
                  </div>
                </div>
              )}

              {/* Address */}
              {patient.user?.address && (
                <div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                    <MapPin className="w-5 h-5 mr-2 text-blue-500" />
                    Địa chỉ
                  </h4>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-gray-700">{patient.user.address}</p>
                  </div>
                </div>
              )}

              {/* Emergency Contact */}
              {patient.emergency_contact && (
                <div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-3">
                    Liên hệ khẩn cấp
                  </h4>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-gray-700">{patient.emergency_contact}</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'appointments' && (
            <div className="space-y-3">
              {appointments.length > 0 ? (
                appointments.map((appt) => (
                  <div key={appt.id} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-medium text-gray-900">
                          {formatDate(appt.appointment_date)} - {appt.appointment_time}
                        </div>
                        <div className="text-sm text-gray-600 mt-1">
                          Bác sĩ: {appt.doctor?.user?.full_name}
                        </div>
                        {appt.reason && (
                          <div className="text-sm text-gray-600 mt-1">
                            Lý do: {appt.reason}
                          </div>
                        )}
                      </div>
                      <Badge text={appt.status} />
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  Chưa có lịch hẹn
                </div>
              )}
            </div>
          )}

          {activeTab === 'records' && (
            <div className="space-y-3">
              {medicalRecords.length > 0 ? (
                medicalRecords.map((record) => (
                  <div key={record.id} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div className="font-medium text-gray-900">
                        {formatDate(record.created_at)}
                      </div>
                      <Badge text={record.diagnosis ? 'Có chẩn đoán' : 'Chưa chẩn đoán'} />
                    </div>
                    <div className="text-sm text-gray-600">
                      Bác sĩ: {record.doctor?.user?.full_name}
                    </div>
                    {record.diagnosis && (
                      <div className="mt-2 text-sm">
                        <span className="font-medium">Chẩn đoán:</span> {record.diagnosis}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  Chưa có bệnh án
                </div>
              )}
            </div>
          )}

          {activeTab === 'prescriptions' && (
            <div className="space-y-3">
              {prescriptions.length > 0 ? (
                prescriptions.map((prescription) => (
                  <div key={prescription.id} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div className="font-medium text-gray-900">
                        {formatDate(prescription.created_at)}
                      </div>
                      <Badge text={prescription.status} />
                    </div>
                    <div className="text-sm text-gray-600">
                      Bác sĩ: {prescription.doctor?.user?.full_name}
                    </div>
                    {prescription.notes && (
                      <div className="mt-2 text-sm text-gray-600">
                        {prescription.notes}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  Chưa có đơn thuốc
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
}