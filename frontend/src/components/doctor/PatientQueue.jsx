// src/components/doctor/PatientQueue.jsx
import React, { useState } from 'react';
import { Search, Users } from 'lucide-react';
import PatientCard from './PatientCard';

const PatientQueue = ({ patients = [], onSelectPatient }) => {
  const [filter, setFilter] = useState('ALL');
  const [searchTerm, setSearchTerm] = useState('');

  const filteredPatients = patients.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filter === 'ALL' || p.status === filter;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="space-y-4">
      <div className="flex gap-3">
        <div className="flex-1 relative">
          <Search className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Tìm bệnh nhân..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <select
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="ALL">Tất cả</option>
          <option value="PENDING">Chờ khám</option>
          <option value="CHECKED_IN">Đã check-in</option>
          <option value="CONFIRMED">Đã xác nhận</option>
        </select>
      </div>
      <div className="grid gap-4">
        {filteredPatients.map((patient) => (
          <PatientCard
            key={patient.id}
            patient={patient}
            onClick={onSelectPatient}
          />
        ))}
      </div>
      {filteredPatients.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <Users className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <p>Không có bệnh nhân nào</p>
        </div>
      )}
    </div>
  );
};

export default PatientQueue;