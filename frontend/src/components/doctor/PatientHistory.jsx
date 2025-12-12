// src/components/doctor/PatientHistory.jsx
import React, { useState } from 'react';
import { FileText, Pill, TestTube, Eye } from 'lucide-react';

const PatientHistory = ({ patientId }) => {
  const [history] = useState({
    medicalRecords: [
      { id: 1, date: '2025-01-05', diagnosis: 'Viêm họng cấp', doctor: 'BS. Nguyễn Văn A' },
      { id: 2, date: '2024-12-20', diagnosis: 'Cảm cúm', doctor: 'BS. Trần Thị B' }
    ],
    prescriptions: [
      { id: 1, date: '2025-01-05', medicines: 'Amoxicillin, Paracetamol', doctor: 'BS. Nguyễn Văn A' }
    ],
    labResults: [
      { id: 1, date: '2024-12-15', test: 'Xét nghiệm máu tổng quát', result: 'Bình thường' }
    ]
  });
  const [activeTab, setActiveTab] = useState('records');

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="border-b">
        <div className="flex">
          <button
            onClick={() => setActiveTab('records')}
            className={`px-6 py-3 font-medium ${activeTab === 'records' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600'}`}
          >
            <FileText className="w-4 h-4 inline mr-2" />
            Bệnh án
          </button>
          <button
            onClick={() => setActiveTab('prescriptions')}
            className={`px-6 py-3 font-medium ${activeTab === 'prescriptions' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600'}`}
          >
            <Pill className="w-4 h-4 inline mr-2" />
            Đơn thuốc
          </button>
          <button
            onClick={() => setActiveTab('labs')}
            className={`px-6 py-3 font-medium ${activeTab === 'labs' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600'}`}
          >
            <TestTube className="w-4 h-4 inline mr-2" />
            Xét nghiệm
          </button>
        </div>
      </div>
      <div className="p-6">
        {activeTab === 'records' && (
          <div className="space-y-4">
            {history.medicalRecords.map(record => (
              <div key={record.id} className="border-l-4 border-blue-500 bg-gray-50 p-4 rounded">
                <div className="flex justify-between items-start mb-2">
                  <div className="font-medium text-gray-900">{record.date}</div>
                  <button className="text-blue-600 hover:text-blue-700">
                    <Eye className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-gray-700 mb-1">{record.diagnosis}</p>
                <p className="text-sm text-gray-600">{record.doctor}</p>
              </div>
            ))}
          </div>
        )}
        {activeTab === 'prescriptions' && (
          <div className="space-y-4">
            {history.prescriptions.map(prescription => (
              <div key={prescription.id} className="border-l-4 border-green-500 bg-gray-50 p-4 rounded">
                <div className="flex justify-between items-start mb-2">
                  <div className="font-medium text-gray-900">{prescription.date}</div>
                  <button className="text-blue-600 hover:text-blue-700">
                    <Eye className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-gray-700 mb-1">{prescription.medicines}</p>
                <p className="text-sm text-gray-600">{prescription.doctor}</p>
              </div>
            ))}
          </div>
        )}
        {activeTab === 'labs' && (
          <div className="space-y-4">
            {history.labResults.map(lab => (
              <div key={lab.id} className="border-l-4 border-purple-500 bg-gray-50 p-4 rounded">
                <div className="flex justify-between items-start mb-2">
                  <div className="font-medium text-gray-900">{lab.date}</div>
                  <button className="text-blue-600 hover:text-blue-700">
                    <Eye className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-gray-700 mb-1">{lab.test}</p>
                <p className="text-sm text-green-600 font-medium">{lab.result}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default PatientHistory;