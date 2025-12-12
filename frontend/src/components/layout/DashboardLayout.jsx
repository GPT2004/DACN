import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from '../common/Header';
import PublicHeader from '../common/PublicHeader';

export default function DashboardLayout({ children, role }) {
  const isPatient = String(role || '').toLowerCase() === 'patient';

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {isPatient ? (
        <PublicHeader />
      ) : (
        <Header showMenu={false} />
      )}

      <div className="flex-1">
        <main className="flex-1 overflow-auto">
          <div className="container mx-auto px-4 py-6">
            {children || <Outlet />}
          </div>
        </main>
      </div>
    </div>
  );
}