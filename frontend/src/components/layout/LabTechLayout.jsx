import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from '../common/Header';

export default function LabTechLayout({ children }) {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Header showMenu={false} />

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
