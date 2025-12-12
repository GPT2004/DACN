import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';
import { formatDate, formatCurrency } from './helpers';

/**
 * Import data from Excel file
 */
export const importFromExcel = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        
        // Get first worksheet
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        
        // Convert to JSON
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
        
        resolve(jsonData);
      } catch (error) {
        reject(error);
      }
    };
    
    reader.onerror = (error) => {
      reject(error);
    };
    
    reader.readAsArrayBuffer(file);
  });
};

/**
 * Parse Excel data for medicines import
 */
export const parseMedicinesFromExcel = (excelData) => {
  if (!excelData || excelData.length < 2) {
    throw new Error('File Excel không có dữ liệu hoặc thiếu header');
  }
  
  const headers = excelData[0];
  const rows = excelData.slice(1);
  
  // Expected headers (case insensitive)
  const expectedHeaders = ['name', 'code', 'description', 'unit', 'price', 'batch_number', 'expiry_date', 'quantity'];
  const normalizedHeaders = headers.map(h => (h || '').toString().toLowerCase().trim());
  
  // Validate required headers
  const hasName = normalizedHeaders.includes('name') || normalizedHeaders.includes('tên thuốc');
  const hasCode = normalizedHeaders.includes('code') || normalizedHeaders.includes('mã thuốc');
  const hasUnit = normalizedHeaders.includes('unit') || normalizedHeaders.includes('đơn vị');
  const hasPrice = normalizedHeaders.includes('price') || normalizedHeaders.includes('giá');
  const hasBatchNumber = normalizedHeaders.includes('batch_number') || normalizedHeaders.includes('số lô');
  const hasExpiryDate = normalizedHeaders.includes('expiry_date') || normalizedHeaders.includes('hạn sử dụng');
  const hasQuantity = normalizedHeaders.includes('quantity') || normalizedHeaders.includes('số lượng');
  
  if (!hasName) {
    throw new Error('Thiếu cột "Tên thuốc" hoặc "name"');
  }
  
  const medicines = [];
  const errors = [];
  
  rows.forEach((row, index) => {
    try {
      const medicine = {};
      
      // Map columns
      headers.forEach((header, colIndex) => {
        const normalizedHeader = (header || '').toString().toLowerCase().trim();
        const value = row[colIndex];
        
        if (normalizedHeader === 'name' || normalizedHeader === 'tên thuốc') {
          medicine.name = value ? value.toString().trim() : '';
        } else if (normalizedHeader === 'code' || normalizedHeader === 'mã thuốc') {
          medicine.code = value ? value.toString().trim() : null;
        } else if (normalizedHeader === 'description' || normalizedHeader === 'mô tả') {
          medicine.description = value ? value.toString().trim() : null;
        } else if (normalizedHeader === 'unit' || normalizedHeader === 'đơn vị') {
          medicine.unit = value ? value.toString().trim() : null;
        } else if (normalizedHeader === 'price' || normalizedHeader === 'giá') {
          const price = parseFloat(value);
          medicine.price = isNaN(price) ? 0 : Math.round(price);
        } else if (normalizedHeader === 'batch_number' || normalizedHeader === 'số lô') {
          medicine.batch_number = value ? value.toString().trim() : null;
        } else if (normalizedHeader === 'expiry_date' || normalizedHeader === 'hạn sử dụng') {
          if (value !== undefined && value !== null && value !== '') {
            // Try to parse date from multiple possible formats:
            // - Excel serial number (number)
            // - JS Date object
            // - String in DD/MM/YYYY or ISO formats
            let dateObj = null;

            try {
              if (typeof value === 'number') {
                // Excel date serial number -> parse_date_code returns {y,m,d,...}
                const parsed = XLSX.SSF.parse_date_code(value);
                if (parsed && parsed.y) {
                  dateObj = new Date(parsed.y, parsed.m - 1, parsed.d);
                }
              } else if (value instanceof Date) {
                dateObj = value;
              } else if (typeof value === 'string') {
                const str = value.trim();
                // Try DD/MM/YYYY
                const dm = str.match(/^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})$/);
                if (dm) {
                  const d = parseInt(dm[1], 10);
                  const m = parseInt(dm[2], 10);
                  const y = parseInt(dm[3], 10);
                  dateObj = new Date(y, m - 1, d);
                } else {
                  // Fallback to Date constructor (handles ISO and some locale formats)
                  const tmp = new Date(str);
                  if (!isNaN(tmp.getTime())) dateObj = tmp;
                }
              }
            } catch (err) {
              dateObj = null;
            }

            // Validate date
            if (!dateObj || isNaN(dateObj.getTime())) {
              throw new Error('Hạn sử dụng không hợp lệ');
            }

            // Normalize to YYYY-MM-DD using local date components to avoid timezone shifts
            const y = dateObj.getFullYear();
            const m = String(dateObj.getMonth() + 1).padStart(2, '0');
            const d = String(dateObj.getDate()).padStart(2, '0');
            medicine.expiry_date = `${y}-${m}-${d}`;
          }
        } else if (normalizedHeader === 'quantity' || normalizedHeader === 'số lượng') {
          const quantity = parseInt(value);
          medicine.quantity = isNaN(quantity) ? 0 : quantity;
        }
      });
      
      // Validate required fields
      if (!medicine.name || medicine.name.trim() === '') {
        errors.push(`Dòng ${index + 2}: Thiếu tên thuốc`);
        return;
      }
      
      if (!medicine.quantity || medicine.quantity <= 0) {
        errors.push(`Dòng ${index + 2}: Số lượng phải lớn hơn 0`);
        return;
      }
      
      if (!medicine.expiry_date) {
        errors.push(`Dòng ${index + 2}: Thiếu hạn sử dụng`);
        return;
      }
      
      // Check if expiry date is in the future
      const expiryDate = new Date(medicine.expiry_date);
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      
      if (expiryDate <= today) {
        errors.push(`Dòng ${index + 2}: Hạn sử dụng phải sau ngày hôm nay`);
        return;
      }
      
      medicines.push(medicine);
      
    } catch (error) {
      errors.push(`Dòng ${index + 2}: Lỗi xử lý dữ liệu - ${error.message}`);
    }
  });
  
  return { medicines, errors };
};

/**
 * Export data to Excel
 */
export const exportToExcel = (data, filename = 'export.xlsx', sheetName = 'Sheet1') => {
  try {
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, sheetName);
    
    const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
    const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
    
    saveAs(blob, filename);
    return true;
  } catch (error) {
    console.error('Error exporting to Excel:', error);
    return false;
  }
};

/**
 * Export data to CSV
 */
export const exportToCSV = (data, filename = 'export.csv') => {
  try {
    const worksheet = XLSX.utils.json_to_sheet(data);
    const csv = XLSX.utils.sheet_to_csv(worksheet);
    
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, filename);
    return true;
  } catch (error) {
    console.error('Error exporting to CSV:', error);
    return false;
  }
};

/**
 * Export table to Excel
 */
export const exportTableToExcel = (tableId, filename = 'table.xlsx') => {
  try {
    const table = document.getElementById(tableId);
    if (!table) {
      console.error('Table not found');
      return false;
    }
    
    const worksheet = XLSX.utils.table_to_sheet(table);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Sheet1');
    
    const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
    const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
    
    saveAs(blob, filename);
    return true;
  } catch (error) {
    console.error('Error exporting table to Excel:', error);
    return false;
  }
};

/**
 * Format appointments data for export
 */
export const formatAppointmentsForExport = (appointments) => {
  return appointments.map(app => ({
    'Mã lịch hẹn': app.id,
    'Bệnh nhân': app.patient?.user?.full_name || '',
    'Bác sĩ': app.doctor?.user?.full_name || '',
    'Ngày khám': formatDate(app.appointment_date),
    'Giờ khám': app.appointment_time,
    'Lý do': app.reason || '',
    'Trạng thái': app.status,
    'Nguồn': app.source,
  }));
};

/**
 * Format patients data for export
 */
export const formatPatientsForExport = (patients) => {
  return patients.map(patient => ({
    'Mã BN': patient.id,
    'Họ tên': patient.user?.full_name || '',
    'Giới tính': patient.gender || '',
    'Ngày sinh': formatDate(patient.user?.dob),
    'Điện thoại': patient.user?.phone || '',
    'Email': patient.user?.email || '',
    'Nhóm máu': patient.blood_type || '',
    'Dị ứng': patient.allergies || '',
  }));
};

/**
 * Format invoices data for export
 */
export const formatInvoicesForExport = (invoices) => {
  return invoices.map(invoice => ({
    'Mã HĐ': invoice.id,
    'Bệnh nhân': invoice.patient?.user?.full_name || invoice.patient?.full_name || '',
    'Ngày tạo': formatDate(invoice.created_at),
    'Tổng phụ': formatCurrency(invoice.subtotal),
    'Thuế': formatCurrency(invoice.tax),
    'Giảm giá': formatCurrency(invoice.discount),
    'Tổng cộng': formatCurrency(invoice.total),
    'Trạng thái': invoice.status,
    'Ngày thanh toán': invoice.paid_at ? formatDate(invoice.paid_at) : '',
  }));
};

/**
 * Format prescriptions data for export
 */
export const formatPrescriptionsForExport = (prescriptions) => {
  return prescriptions.map(prescription => ({
    'Mã đơn': prescription.id,
    'Bệnh nhân': prescription.patient?.user?.full_name || prescription.patient?.full_name || '',
    'Bác sĩ': prescription.doctor?.user?.full_name || '',
    'Ngày kê': formatDate(prescription.created_at),
    'Tổng tiền': formatCurrency(prescription.total_amount),
    'Trạng thái': prescription.status,
  }));
};

/**
 * Format medical records data for export
 */
export const formatMedicalRecordsForExport = (records) => {
  return records.map(record => ({
    'Mã BA': record.id,
    'Bệnh nhân': record.patient?.user?.full_name || record.patient?.full_name || '',
    'Bác sĩ': record.doctor?.user?.full_name || '',
    'Ngày khám': formatDate(record.created_at),
    'Chẩn đoán': record.diagnosis || '',
    'Ghi chú': record.notes || '',
  }));
};

/**
 * Format medicines data for export
 */
export const formatMedicinesForExport = (medicines) => {
  return medicines.map(medicine => ({
    'Mã thuốc': medicine.code || '',
    'Tên thuốc': medicine.name,
    'Đơn vị': medicine.unit,
    'Giá': formatCurrency(medicine.price),
    'Tồn kho': medicine.total_stock || 0,
  }));
};