import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from '../common/Header';

export default function ReceptionistLayout({ children }) {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Header showMenu={false} />

      <main className="flex-1 overflow-auto">
        <div className="container mx-auto px-4 py-6">
          {children || <Outlet />}
        </div>
      </main>
    </div>
  );
}
