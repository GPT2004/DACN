const invoiceService = require('./invoices.service');
const PDFService = require('../../utils/pdf');
const { successResponse } = require('../../utils/response');

class InvoiceController {
  async createInvoice(req, res, next) {
    try {
      const invoice = await invoiceService.createInvoice(req.validatedBody);
      return successResponse(res, invoice, 'Invoice created successfully', 201);
    } catch (error) {
      next(error);
    }
  }

  async getInvoices(req, res, next) {
    try {
      const { patient_id, status, start_date, end_date, page = 1, limit = 10 } = req.query;
      const result = await invoiceService.getInvoices({
        patient_id,
        status,
        start_date,
        end_date,
        page: parseInt(page),
        limit: parseInt(limit),
      }, req.user);
      return successResponse(res, result, 'Invoices retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getInvoicesByPatient(req, res, next) {
    try {
      const result = await invoiceService.getInvoices({
        page: 1,
        limit: 100,
      }, req.user);
      return successResponse(res, result, 'Patient invoices retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async getInvoiceById(req, res, next) {
    try {
      const invoice = await invoiceService.getInvoiceById(req.params.id, req.user);
      return successResponse(res, invoice, 'Invoice retrieved successfully');
    } catch (error) {
      next(error);
    }
  }

  async updateInvoice(req, res, next) {
    try {
      const invoice = await invoiceService.updateInvoice(req.params.id, req.validatedBody);
      return successResponse(res, invoice, 'Invoice updated successfully');
    } catch (error) {
      next(error);
    }
  }

  async payInvoice(req, res, next) {
    try {
      const { payment_method, paid_amount } = req.body;
      const result = await invoiceService.payInvoice(req.params.id, payment_method, paid_amount);
      // result: { invoice, change }
      return successResponse(res, result, 'Invoice paid successfully');
    } catch (error) {
      next(error);
    }
  }

  async createFromPrescription(req, res, next) {
    try {
      const { prescription_id } = req.body;
      const invoice = await invoiceService.createInvoiceFromPrescription(prescription_id, req.user?.id);
      return successResponse(res, invoice, 'Invoice created from prescription', 201);
    } catch (error) {
      next(error);
    }
  }

  async downloadInvoicePDF(req, res, next) {
    try {
      const invoice = await invoiceService.getInvoiceById(req.params.id, req.user);
      const uploadsDir = require('path').join(__dirname, '../../../uploads/invoices');
      const fs = require('fs');
      if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
      const outputPath = require('path').join(uploadsDir, `invoice-${invoice.id}.pdf`);
      await PDFService.generateInvoicePDF(invoice, outputPath);
      return res.download(outputPath);
    } catch (error) {
      next(error);
    }
  }

  async refundInvoice(req, res, next) {
    try {
      const { reason } = req.body;
      const invoice = await invoiceService.refundInvoice(req.params.id, reason);
      return successResponse(res, invoice, 'Invoice refunded successfully');
    } catch (error) {
      next(error);
    }
  }

  async deleteInvoice(req, res, next) {
    try {
      await invoiceService.deleteInvoice(req.params.id);
      return successResponse(res, null, 'Invoice deleted successfully');
    } catch (error) {
      next(error);
    }
  }

  async getRevenueSummary(req, res, next) {
    try {
      const { start_date, end_date } = req.query;
      const summary = await invoiceService.getRevenueSummary(start_date, end_date);
      return successResponse(res, summary, 'Revenue summary retrieved successfully');
    } catch (error) {
      next(error);
    }
  }
}

module.exports = new InvoiceController();