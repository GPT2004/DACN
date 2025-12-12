import React from 'react';
import PublicHeader from '../common/PublicHeader';

export default function PatientLayout({ children, title, description }) {
  return (
    <>
      <PublicHeader />
      <main className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          {title && (
            <div className="mb-8">
              <h1 className="text-4xl font-bold text-gray-900 mb-2">{title}</h1>
              {description && <p className="text-gray-600">{description}</p>}
            </div>
          )}

          {/* Content */}
          {children}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-300 py-8 mt-16">
        <div className="max-w-5xl mx-auto px-4 text-center">
          <p>&copy; 2025 Minh An Clinic. Tất cả quyền được bảo lưu.</p>
        </div>
      </footer>
    </>
  );
}
