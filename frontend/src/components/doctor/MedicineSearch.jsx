// src/components/doctor/MedicineSearch.jsx
import React, { useState } from 'react';
import { Search } from 'lucide-react';

const MedicineSearch = ({ onSelect }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [medicines] = useState([
    { id: 1, name: 'Amoxicillin 500mg', type: 'Kháng sinh', unit: 'Viên', price: 5000 },
    { id: 2, name: 'Paracetamol 500mg', type: 'Giảm đau', unit: 'Viên', price: 2000 },
    { id: 3, name: 'Vitamin C 1000mg', type: 'Vitamin', unit: 'Viên', price: 3000 },
    { id: 4, name: 'Omeprazole 20mg', type: 'Tiêu hóa', unit: 'Viên', price: 8000 },
  ]);

  const filteredMedicines = medicines.filter(m =>
    m.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="bg-white rounded-lg border border-gray-300">
      <div className="p-3 border-b">
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Tìm thuốc..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>
      <div className="max-h-64 overflow-y-auto">
        {filteredMedicines.map(medicine => (
          <div
            key={medicine.id}
            onClick={() => onSelect(medicine)}
            className="p-3 hover:bg-gray-50 cursor-pointer border-b last:border-b-0"
          >
            <div className="font-medium text-gray-900">{medicine.name}</div>
            <div className="flex justify-between items-center mt-1">
              <span className="text-xs text-gray-500">{medicine.type}</span>
              <span className="text-sm font-medium text-blue-600">
                {medicine.price.toLocaleString('vi-VN')} ₫/{medicine.unit}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MedicineSearch;