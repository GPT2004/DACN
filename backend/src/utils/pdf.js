const PDFDocument = require('pdfkit');
const fs = require('fs');

class PDFService {
  generatePrescriptionPDF(prescription, outputPath) {
    return new Promise((resolve, reject) => {
      try {
        const doc = new PDFDocument({ margin: 50 });
        const stream = fs.createWriteStream(outputPath);

        doc.pipe(stream);

        // Header
        doc.fontSize(20).text('ĐơN THUỐC', { align: 'center' });
        doc.moveDown();

        // Patient info
        doc.fontSize(12);
        doc.text(`Bệnh nhân: ${prescription.patient.user.full_name}`);
        doc.text(`Ngày: ${new Date(prescription.created_at).toLocaleDateString('vi-VN')}`);
        doc.text(`Bác sĩ: ${prescription.doctor.user.full_name}`);
        doc.moveDown();

        // Table header
        doc.fontSize(10);
        const tableTop = doc.y;
        doc.text('STT', 50, tableTop);
        doc.text('Tên thuốc', 100, tableTop);
        doc.text('SL', 300, tableTop);
        doc.text('Đơn vị', 350, tableTop);
        doc.text('Cách dùng', 420, tableTop);

        doc.moveTo(50, tableTop + 20).lineTo(550, tableTop + 20).stroke();

        // Items
        let yPos = tableTop + 30;
        prescription.items.forEach((item, index) => {
          doc.text(index + 1, 50, yPos);
          doc.text(item.medicine_name, 100, yPos);
          doc.text(item.quantity, 300, yPos);
          doc.text(item.unit, 350, yPos);
          doc.text(item.instructions || '', 420, yPos, { width: 130 });
          yPos += 25;
        });

        doc.moveDown();
        doc.fontSize(12);
        doc.text(`Tổng tiền: ${prescription.total_amount.toLocaleString('vi-VN')} VNĐ`, { align: 'right' });

        // Footer
        doc.moveDown(3);
        doc.fontSize(10);
        doc.text('Chữ ký bác sĩ', 400, doc.y, { align: 'center' });

        doc.end();

        stream.on('finish', () => resolve(outputPath));
        stream.on('error', reject);
      } catch (error) {
        reject(error);
      }
    });
  }

  generateInvoicePDF(invoice, outputPath) {
    return new Promise((resolve, reject) => {
      try {
        const doc = new PDFDocument({ margin: 50 });
        const stream = fs.createWriteStream(outputPath);

        doc.pipe(stream);

        // Header
        doc.fontSize(20).text('HÓA ĐƠN THANH TOÁN', { align: 'center' });
        doc.moveDown();

        // Info
        doc.fontSize(12);
        doc.text(`Mã HĐ: INV-${invoice.id}`);
        doc.text(`Bệnh nhân: ${invoice.patient.user.full_name}`);
        doc.text(`Ngày: ${new Date(invoice.created_at).toLocaleDateString('vi-VN')}`);
        doc.moveDown();

        // Items
        doc.fontSize(10);
        const tableTop = doc.y;
        doc.text('Dịch vụ', 50, tableTop);
        doc.text('Số lượng', 300, tableTop);
        doc.text('Đơn giá', 400, tableTop);
        doc.text('Thành tiền', 480, tableTop);

        doc.moveTo(50, tableTop + 20).lineTo(550, tableTop + 20).stroke();

        let yPos = tableTop + 30;
        invoice.items.forEach((item) => {
          doc.text(item.description, 50, yPos);
          doc.text(item.quantity || 1, 300, yPos);
          doc.text(item.unit_price.toLocaleString('vi-VN'), 400, yPos);
          doc.text(item.amount.toLocaleString('vi-VN'), 480, yPos);
          yPos += 25;
        });

        // Total
        doc.moveTo(50, yPos).lineTo(550, yPos).stroke();
        yPos += 10;

        doc.fontSize(12);
        doc.text(`Tổng phụ: ${invoice.subtotal.toLocaleString('vi-VN')} VNĐ`, 350, yPos);
        yPos += 20;
        doc.text(`Thuế: ${invoice.tax.toLocaleString('vi-VN')} VNĐ`, 350, yPos);
        yPos += 20;
        doc.text(`Giảm giá: ${invoice.discount.toLocaleString('vi-VN')} VNĐ`, 350, yPos);
        yPos += 20;
        doc.fontSize(14).text(`TỔNG CỘNG: ${invoice.total.toLocaleString('vi-VN')} VNĐ`, 350, yPos);

        doc.end();

        stream.on('finish', () => resolve(outputPath));
        stream.on('error', reject);
      } catch (error) {
        reject(error);
      }
    });
  }

  generateMedicalRecordPDF(record, outputPath) {
    return new Promise((resolve, reject) => {
      try {
        const doc = new PDFDocument({ margin: 50 });
        const stream = fs.createWriteStream(outputPath);

        doc.pipe(stream);

        doc.fontSize(20).text('BỆNH ÁN', { align: 'center' });
        doc.moveDown();

        doc.fontSize(12);
        doc.text(`Bệnh nhân: ${record.patient.user.full_name}`);
        doc.text(`Ngày khám: ${new Date(record.created_at).toLocaleDateString('vi-VN')}`);
        doc.text(`Bác sĩ: ${record.doctor.user.full_name}`);
        doc.moveDown();

        doc.fontSize(14).text('Chẩn đoán:', { underline: true });
        doc.fontSize(12).text(record.diagnosis || 'Không có');
        doc.moveDown();

        doc.fontSize(14).text('Ghi chú:', { underline: true });
        doc.fontSize(12).text(record.notes || 'Không có');

        doc.end();

        stream.on('finish', () => resolve(outputPath));
        stream.on('error', reject);
      } catch (error) {
        reject(error);
      }
    });
  }
}

module.exports = new PDFService();