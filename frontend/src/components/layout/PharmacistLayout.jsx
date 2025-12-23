/* eslint-disable no-console */
import React, { useState, useEffect } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import Header from '../common/Header';
import { Package, Pill, FileText } from 'lucide-react';
import { invoiceService } from '../../services/invoiceService';
import { prescriptionService } from '../../services/prescriptionService';

export default function PharmacistLayout({ children }) {
  const location = useLocation();
  const [pendingCount, setPendingCount] = useState(0);
  
  useEffect(() => {
    loadPendingCount();
    // Refresh every 30 seconds
    const interval = setInterval(loadPendingCount, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadPendingCount = async () => {
    try {
      const res = await invoiceService.getInvoices({ status: 'PAID', page: 1, limit: 1000 });
      let items = [];
      if (res && res.data && res.data.invoices) {
        items = res.data.invoices;
      } else if (res && res.invoices) {
        items = res.invoices;
      } else if (Array.isArray(res)) {
        items = res;
      }

      // Count invoices with non-DISPENSED prescriptions
      let pendingCount = 0;
      for (const inv of items) {
        if (inv.prescription_id) {
          try {
            const presRes = await prescriptionService.getPrescriptionById(inv.prescription_id);
            let presData = presRes?.data || presRes;
            if (presData && presData.data) presData = presData.data;
            if (presData && presData.prescription) presData = presData.prescription;
            if (presData?.status !== 'DISPENSED') {
              pendingCount++;
            }
          } catch (e) {
            // ignore
          }
        }
      }
      setPendingCount(pendingCount);
    } catch (err) {
      console.error('Failed to load pending count', err);
    }
  };
  
  const isActive = (path) => location.pathname === path ? 'border-b-2 border-green-600 text-green-600' : 'text-gray-600 hover:text-gray-900';
  
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Header showMenu={false} />
      
      {/* Pharmacist Navigation */}
      <nav className="bg-white border-b border-gray-200 sticky top-16 z-30">
        <div className="container mx-auto px-4">
          <div className="flex gap-8">
            <Link 
              to="/pharmacist" 
              className={`px-4 py-3 font-medium flex items-center gap-2 border-b-2 border-transparent transition ${isActive('/pharmacist')}`}
            >
              <Package size={18} />
              Trang chủ
            </Link>
            <Link 
              to="/pharmacist/invoices-ready" 
              className={`px-4 py-3 font-medium flex items-center gap-2 border-b-2 border-transparent transition relative ${isActive('/pharmacist/invoices-ready')}`}
            >
              <FileText size={18} />
              Đơn thuốc
              {pendingCount > 0 && (
                <span className="absolute top-1 right-0 bg-red-600 text-white text-xs font-bold rounded-full h-5 w-5 flex items-center justify-center">
                  {pendingCount > 99 ? '99+' : pendingCount}
                </span>
              )}
            </Link>
            <Link 
              to="/pharmacist/medicines" 
              className={`px-4 py-3 font-medium flex items-center gap-2 border-b-2 border-transparent transition ${isActive('/pharmacist/medicines')}`}
            >
              <Pill size={18} />
              Quản lý thuốc
            </Link>
          </div>
        </div>
      </nav>

      <div className="flex-1 overflow-hidden">
        <main className="flex-1 overflow-auto">
          <div className="container mx-auto px-4 py-6">
            {children || <Outlet />}
          </div>
        </main>
      </div>
    </div>
  );
}
