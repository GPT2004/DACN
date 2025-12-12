import React from 'react';

export default function PatientPageHeader({ title, description }) {
  return (
    <div className="bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-emerald-200 py-6">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">{title}</h1>
        {description && <p className="text-gray-600">{description}</p>}
      </div>
    </div>
  );
}
