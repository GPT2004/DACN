import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import PharmacistLayout from '../../components/layout/PharmacistLayout';
import { medicineService } from '../../services/medicineService';
import { invoiceService } from '../../services/invoiceService';
import { prescriptionService } from '../../services/prescriptionService';
import { Package, AlertCircle, TrendingDown, ShoppingCart } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function PharmacistDashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalMedicines: 0,
    lowStockItems: 0,
    pendingPrescriptions: 0,
    expiringMedicines: 0
  });
  const [recentPrescriptions, setRecentPrescriptions] = useState([]);
  const [lowStockAlerts, setLowStockAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);

      // Fetch medicines
      const medsRes = await medicineService.getMedicines({ limit: 1000 });
      let medsList = [];
      if (medsRes && medsRes.data && medsRes.data.medicines) {
        medsList = medsRes.data.medicines;
      } else if (medsRes && medsRes.medicines) {
        medsList = medsRes.medicines;
      } else if (Array.isArray(medsRes)) {
        medsList = medsRes;
      }

      // Fetch stocks
      const stocksRes = await medicineService.getAllStocks({ limit: 10000 });
      let stocksList = [];
      if (stocksRes && stocksRes.data && stocksRes.data.stocks) {
        stocksList = stocksRes.data.stocks;
      } else if (stocksRes && stocksRes.stocks) {
        stocksList = stocksRes.stocks;
      } else if (Array.isArray(stocksRes)) {
        stocksList = stocksRes;
      }

      // Calculate stock per medicine
      const stockMap = {};
      const expiringList = [];
      stocksList.forEach(s => {
        const medId = s.medicine_id;
        if (!stockMap[medId]) {
          stockMap[medId] = 0;
        }
        const isExpired = s.expiry_date && new Date(s.expiry_date) <= new Date();
        const isExpiring = s.expiry_date && new Date(s.expiry_date) <= new Date(Date.now() + 30 * 24 * 60 * 60 * 1000);
        
        if (!isExpired) {
          stockMap[medId] += s.quantity || 0;
        }
        if (isExpiring && !isExpired) {
          const med = medsList.find(m => m.id === medId);
          expiringList.push({
            id: med?.id,
            name: med?.name,
            expiry_date: s.expiry_date,
            quantity: s.quantity,
            batch_number: s.batch_number
          });
        }
      });

      // Calculate stats
      const lowStocks = medsList.filter(m => (stockMap[m.id] || 0) < 30);
      const outOfStock = medsList.filter(m => (stockMap[m.id] || 0) === 0);

      setStats({
        totalMedicines: medsList.length,
        lowStockItems: lowStocks.length + outOfStock.length,
        pendingPrescriptions: 0, // will load from API
        expiringMedicines: expiringList.length
      });

      // Get low stock alerts
      setLowStockAlerts(
        lowStocks.slice(0, 5).map(m => ({
          id: m.id,
          name: m.name,
          stock: stockMap[m.id] || 0,
          unit: m.unit || 'viên'
        }))
      );

      // Fetch recent invoices (paid, ready for dispensing)
      const invRes = await invoiceService.getInvoices({ 
        status: 'PAID',
        page: 1, 
        limit: 5
      });
      let invList = [];
      if (invRes && invRes.data && invRes.data.invoices) {
        invList = invRes.data.invoices;
      } else if (invRes && invRes.invoices) {
        invList = invRes.invoices;
      } else if (Array.isArray(invRes)) {
        invList = invRes;
      }

      // Enrich invoices with prescription status
      const enrichedInvoices = await Promise.all(invList.slice(0, 2).map(async (inv) => {
        let prescStatus = null;
        if (inv.prescription_id) {
          try {
            const presRes = await prescriptionService.getPrescriptionById(inv.prescription_id);
            let presData = presRes?.data || presRes;
            if (presData && presData.data) presData = presData.data;
            if (presData && presData.prescription) presData = presData.prescription;
            prescStatus = presData?.status;
          } catch (e) {
            // ignore
          }
        }
        return {
          ...inv,
          prescription_status: prescStatus
        };
      }));

      setRecentPrescriptions(enrichedInvoices);
      setStats(prev => ({
        ...prev,
        pendingPrescriptions: invList.length
      }));
    } catch (err) {
      console.error('Failed to load dashboard data', err);
    } finally {
      setLoading(false);
    }
  };

  const navigate = useNavigate();

  const goToMedicines = (filter) => {
    if (filter) navigate(`/pharmacist/medicines?filter=${filter}`);
    else navigate('/pharmacist/medicines');
  };

  const goToPrescriptions = (status) => {
    if (status) navigate(`/pharmacist/prescriptions?status=${status}`);
    else navigate('/pharmacist/prescriptions');
  };

  return (
    <PharmacistLayout>
      <div className="space-y-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Chào mừng, {user?.full_name}
          </h1>
          <p className="text-gray-600 mt-2">Quản lý thuốc và đơn thuốc cho Phòng khám đa khoa</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div onClick={() => goToMedicines()} className="bg-white rounded-lg shadow p-6 cursor-pointer hover:shadow-md transition">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Tổng loại thuốc</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats.totalMedicines}</p>
              </div>
              <Package className="text-blue-500" size={40} />
            </div>
          </div>

          <div onClick={() => goToMedicines('low')} className="bg-white rounded-lg shadow p-6 cursor-pointer hover:shadow-md transition">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Hết/Sắp hết</p>
                <p className="text-3xl font-bold text-orange-600 mt-2">{stats.lowStockItems}</p>
              </div>
              <TrendingDown className="text-orange-500" size={40} />
            </div>
          </div>

          <div onClick={() => goToPrescriptions('pending')} className="bg-white rounded-lg shadow p-6 cursor-pointer hover:shadow-md transition">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Đơn chờ xử lý</p>
                <p className="text-3xl font-bold text-purple-600 mt-2">{stats.pendingPrescriptions}</p>
              </div>
              <ShoppingCart className="text-purple-500" size={40} />
            </div>
          </div>

          <div onClick={() => goToMedicines('expiring')} className="bg-white rounded-lg shadow p-6 cursor-pointer hover:shadow-md transition">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Sắp hết hạn</p>
                <p className="text-3xl font-bold text-red-600 mt-2">{stats.expiringMedicines}</p>
              </div>
              <AlertCircle className="text-red-500" size={40} />
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Đơn thuốc gần đây</h2>
            <div className="space-y-3">
              {recentPrescriptions.length > 0 ? (
                recentPrescriptions.map(inv => (
                  <div key={inv.id} className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">Đơn #{inv.id} - {inv.patient?.user?.full_name || 'N/A'}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {Array.isArray(inv.items) ? inv.items.length : 0} dịch vụ/thuốc • {
                        inv.prescription_status === 'DISPENSED' ? '✓ Đã bốc' : 'Chưa bốc'
                      }
                    </p>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500">Không có đơn thuốc nào</p>
              )}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Cảnh báo tồn kho</h2>
            <div className="space-y-3">
              {lowStockAlerts.length > 0 ? (
                lowStockAlerts.map(alert => (
                  <div key={alert.id} className={`p-4 rounded-lg border ${alert.stock === 0 ? 'bg-red-50 border-red-200' : 'bg-yellow-50 border-yellow-200'}`}>
                    <p className={`text-sm ${alert.stock === 0 ? 'text-red-700' : 'text-yellow-700'}`}>
                      {alert.name} - Còn {alert.stock} {alert.unit}
                    </p>
                    <p className={`text-xs mt-1 ${alert.stock === 0 ? 'text-red-600' : 'text-yellow-600'}`}>
                      {alert.stock === 0 ? 'Hết hàng, cần đặt ngay' : 'Sắp hết, cần đặt hàng'}
                    </p>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500">Tồn kho bình thường</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </PharmacistLayout>
  );
}
