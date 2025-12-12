// src/components/doctor/PatientCard.jsx
import React from 'react';
import { User, Clock, Stethoscope, Eye } from 'lucide-react';

const PatientCard = ({ patient, onClick }) => {
  const getStatusColor = (status) => {
    const colors = {
      CHECKED_IN: 'bg-green-100 text-green-800',
      CONFIRMED: 'bg-blue-100 text-blue-800',
      PENDING: 'bg-yellow-100 text-yellow-800',
      COMPLETED: 'bg-gray-100 text-gray-800'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div
      className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow cursor-pointer border-l-4 border-blue-500"
      onClick={() => onClick(patient)}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
            <User className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">{patient.name}</h3>
            <p className="text-sm text-gray-600">
              {patient.age} tuổi • {patient.gender}
            </p>
          </div>
        </div>
        <span className={`px-2 py-1 text-xs font-medium rounded ${getStatusColor(patient.status)}`}>
          {patient.status}
        </span>
      </div>

      <div className="space-y-2 text-sm">
        <div className="flex items-center text-gray-600">
          <Clock className="w-4 h-4 mr-2" />
          {patient.time}
        </div>
        <div className="flex items-center text-gray-600">
          <Stethoscope className="w-4 h-4 mr-2" />
          {patient.reason}
        </div>
      </div>
      <div className="mt-3 pt-3 border-t flex gap-2">
        <button className="flex-1 px-3 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition">
          Khám bệnh
        </button>
        <button className="px-3 py-2 border border-gray-300 text-gray-700 text-sm rounded hover:bg-gray-50 transition">
          <Eye className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default PatientCard;