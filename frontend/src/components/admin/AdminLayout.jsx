import React, { useState } from 'react';
import { Menu, X } from 'lucide-react';
import Sidebar from './Sidebar';
import Topbar from './Topbar';

export default function AdminLayout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="h-screen bg-gray-50 flex overflow-hidden">
      {/* Sidebar - Desktop Fixed */}
      <div
        className={`
          fixed left-0 top-0 h-screen bg-white border-r border-gray-200 
          transition-all duration-300 z-40
          ${sidebarOpen ? 'w-64' : 'w-0 overflow-hidden'}
          md:relative md:w-64 md:block
        `}
      >
        <Sidebar />
      </div>

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 md:hidden z-30"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Mobile Menu Toggle */}
        <div className="flex items-center px-4 py-3 bg-white border-b border-gray-200 md:hidden z-20">
          <button
            onClick={toggleSidebar}
            className="p-2 hover:bg-gray-100 rounded-md transition"
          >
            {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Topbar */}
        <Topbar />

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto">
          <div className="p-4 md:p-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
