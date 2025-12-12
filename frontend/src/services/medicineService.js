// src/services/medicineService.js
import api from './api';

export const medicineService = {
  // Get medicines
  getMedicines: async (params = {}) => {
    return await api.get('/medicines', { params });
  },

  // Get medicine by ID
  getMedicineById: async (id) => {
    return await api.get(`/medicines/${id}`);
  },

  // Create medicine (Admin/Pharmacist)
  createMedicine: async (data) => {
    return await api.post('/medicines', data);
  },

  // Update medicine (Admin/Pharmacist)
  updateMedicine: async (id, data) => {
    return await api.put(`/medicines/${id}`, data);
  },

  // Delete medicine (Admin)
  deleteMedicine: async (id) => {
    return await api.delete(`/medicines/${id}`);
  },

  // Get all stocks
  getAllStocks: async (params = {}) => {
    return await api.get('/medicines/stocks/all', { params });
  },

  // Get stock summary
  getStockSummary: async () => {
    return await api.get('/medicines/stocks/summary');
  },

  // Get low stock alerts
  getLowStockAlerts: async () => {
    return await api.get('/medicines/stocks/low-stock');
  },

  // Get expiring medicines
  getExpiringMedicines: async (days = 30) => {
    return await api.get('/medicines/stocks/expiring', { params: { days } });
  },

  // Get stock by ID
  getStockById: async (id) => {
    return await api.get(`/medicines/stocks/${id}`);
  },

  // Get total stock for a medicine
  getStockByMedicineId: async (medicineId) => {
    return await api.get(`/medicines/${medicineId}/stock`);
  },

  // Create stock
  createStock: async (data) => {
    return await api.post('/medicines/stocks', data);
  },

  // Update stock
  updateStock: async (id, data) => {
    return await api.put(`/medicines/stocks/${id}`, data);
  },

  // Import medicines from Excel
  importMedicines: async (medicinesData) => {
    return await api.post('/medicines/import', { medicines: medicinesData });
  },
};

// Named exports (aliases) so existing imports like `import { getAllMedicines } from '.../medicineService'` work
export const getAllMedicines = medicineService.getMedicines;
export const getMedicineById = medicineService.getMedicineById;
export const createMedicine = medicineService.createMedicine;
export const updateMedicine = medicineService.updateMedicine;
export const deleteMedicine = medicineService.deleteMedicine;

export const getAllStocks = medicineService.getAllStocks;
export const getStockSummary = medicineService.getStockSummary;
export const getLowStockAlerts = medicineService.getLowStockAlerts;
export const getExpiringMedicines = medicineService.getExpiringMedicines;
export const getStockById = medicineService.getStockById;
export const createStock = medicineService.createStock;
export const updateStock = medicineService.updateStock;
export const deleteStock = medicineService.deleteStock;
export const importMedicines = medicineService.importMedicines;

export default medicineService;
