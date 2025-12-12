// src/components/receptionist/PatientSearch.jsx
import React, { useState } from 'react';
import { Search, User, Phone } from 'lucide-react';
import { mockPatients } from './mockData';

const PatientSearch = ({ onSelect }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  const handleSearch = (value) => {
    setQuery(value);
    if (value.length > 2) {
      const filtered = mockPatients.filter(p =>
        p.name.toLowerCase().includes(value.toLowerCase()) ||
        p.phone.includes(value)
      );
      setResults(filtered);
    } else {
      setResults([]);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-4">
      <div className="relative">
        <Search className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
        <input
          type="text"
          placeholder="Tìm bệnh nhân (tên/số điện thoại)..."
          className="w-full pl-10 pr-4 py-3 border rounded-lg"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
        />
      </div>
      {results.length > 0 && (
        <div className="mt-3 border rounded-lg max-h-64 overflow-y-auto">
          {results.map(patient => (
            <div
              key={patient.id}
              onClick={() => onSelect(patient)}
              className="p-3 hover:bg-gray-50 cursor-pointer border-b last:border-b-0"
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                  <User className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <p className="font-medium">{patient.name}</p>
                  <p className="text-sm text-gray-600 flex items-center gap-1">
                    <Phone className="w-3 h-3" />
                    {patient.phone}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default PatientSearch;