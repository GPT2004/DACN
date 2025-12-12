// src/services/invoiceService.js
import api from './api';

export const invoiceService = {
  // Get invoices
  getInvoices: async (params = {}) => {
    return await api.get('/invoices', { params });
  },

  // Get invoice by ID
  getInvoiceById: async (id) => {
    return await api.get(`/invoices/${id}`);
  },

  // Get revenue summary
  getRevenueSummary: async (startDate, endDate) => {
    return await api.get('/invoices/reports/revenue', {
      params: { start_date: startDate, end_date: endDate },
    });
  },

  // Create invoice (Receptionist/Admin)
  createInvoice: async (data) => {
    return await api.post('/invoices', data);
  },

  // Update invoice (Receptionist/Admin)
  updateInvoice: async (id, data) => {
    return await api.put(`/invoices/${id}`, data);
  },

  // Pay invoice (Receptionist/Admin)
  payInvoice: async (id, paymentMethod, paidAmount) => {
    return await api.patch(`/invoices/${id}/pay`, {
      payment_method: paymentMethod,
      paid_amount: paidAmount,
    });
  },

  // Create invoice from prescription (Receptionist/Admin)
  createFromPrescription: async (prescriptionId) => {
    return await api.post('/invoices/from-prescription', { prescription_id: prescriptionId });
  },

  // Refund invoice (Admin)
  refundInvoice: async (id, reason) => {
    return await api.patch(`/invoices/${id}/refund`, { reason });
  },

  // Delete invoice (Admin)
  deleteInvoice: async (id) => {
    return await api.delete(`/invoices/${id}`);
  },

  // Download invoice PDF
  downloadInvoicePDF: async (id) => {
    return await api.get(`/invoices/${id}/pdf`, {
      responseType: 'blob',
    });
  },
};