const nodemailer = require('nodemailer');
const logger = require('./logger');

class EmailService {
  constructor() {
    this.transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: parseInt(process.env.SMTP_PORT),
      secure: false,
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASSWORD,
      },
    });
  }

  // Normalize different time shapes (Date, string, {hours,minutes}) to HH:mm
  formatTime(t) {
    try {
      if (!t) return '';
      if (t instanceof Date) {
        const iso = t.toISOString();
        return `${iso.substring(11, 13)}:${iso.substring(14, 16)}`;
      }
      if (typeof t === 'number') {
        const ms = t < 1e12 ? t * 1000 : t; // handle seconds vs milliseconds
        const d = new Date(ms);
        if (!isNaN(d.getTime())) {
          const iso = d.toISOString();
          return `${iso.substring(11, 13)}:${iso.substring(14, 16)}`;
        }
      }
      if (typeof t === 'string') {
        const m = t.match(/^(\d{1,2}):(\d{2})(?::\d{2})?$/);
        if (m) return `${m[1].padStart(2, '0')}:${m[2]}`;
        const d = new Date(t);
        if (!isNaN(d.getTime())) {
          const iso = d.toISOString();
          return `${iso.substring(11, 13)}:${iso.substring(14, 16)}`;
        }
        return t;
      }
      if (typeof t === 'object' && t.hours !== undefined && t.minutes !== undefined) {
        return `${String(t.hours).padStart(2, '0')}:${String(t.minutes).padStart(2, '0')}`;
      }
      return String(t);
    } catch (e) {
      try { return String(t); } catch { return ''; }
    }
  }

  async sendEmail({ to, subject, html, attachments = [] }) {
    try {
      const info = await this.transporter.sendMail({
        from: `"Clinic Management" <${process.env.SMTP_USER}>`,
        to,
        subject,
        html,
        attachments,
      });

      logger.info(`Email sent: ${info.messageId}`);
      return { success: true, messageId: info.messageId };
    } catch (error) {
      logger.error('Email send error:', error);
      throw new Error('Failed to send email');
    }
  }

  async sendAppointmentConfirmation(appointment, patientEmail) {
    const confirmUrl = appointment.confirm_url || '#';
    const isConfirmed = appointment.patient_confirmed;
    const html = `
      <h2>${isConfirmed ? 'Lịch hẹn đã được xác nhận' : 'Xác nhận lịch hẹn'}</h2>
      <p>Kính gửi bệnh nhân,</p>
      <p>Lịch hẹn của bạn ${isConfirmed ? 'đã được xác nhận' : 'đã được đặt'} với thông tin sau:</p>
      <ul>
        <li><strong>Bác sĩ:</strong> ${appointment.doctor.user.full_name}</li>
        <li><strong>Ngày khám:</strong> ${new Date(appointment.appointment_date).toLocaleDateString('vi-VN')}</li>
        <li><strong>Giờ khám:</strong> ${this.formatTime(appointment.appointment_time)}</li>
        <li><strong>Lý do khám:</strong> ${appointment.reason || 'Không có'}</li>
      </ul>
      ${!isConfirmed ? `
      <p><strong>Vui lòng xác nhận lịch hẹn bằng cách nhấn vào nút dưới đây:</strong></p>
      <p style="text-align: center; margin: 20px 0;">
        <a href="${confirmUrl}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">Xác nhận lịch hẹn</a>
      </p>
      <p>Nếu không xác nhận trong vòng 15 phút, lịch hẹn sẽ bị huỷ tự động. Vui lòng đến sớm trước 15 phút.</p>
      ` : '<p>Lịch hẹn của bạn đã được xác nhận. Vui lòng đến đúng giờ.</p>'}
      <p>Nếu cần thay đổi, vui lòng liên hệ phòng khám.</p>
      <p>Trân trọng,<br>Phòng khám</p>
    `;

    return this.sendEmail({
      to: patientEmail,
      subject: isConfirmed ? 'Lịch hẹn đã được xác nhận' : 'Xác nhận lịch hẹn khám bệnh',
      html,
    });
  }

  async sendAppointmentCancellation(appointment, patientEmail, reason) {
    const html = `
      <h2>Thông báo huỷ lịch hẹn</h2>
      <p>Kính gửi bệnh nhân,</p>
      <p>Lịch hẹn của bạn đã bị huỷ do ${reason || 'không xác nhận trong thời gian quy định'}.</p>
      <ul>
        <li><strong>Bác sĩ:</strong> ${appointment.doctor.user.full_name}</li>
        <li><strong>Ngày khám:</strong> ${new Date(appointment.appointment_date).toLocaleDateString('vi-VN')}</li>
        <li><strong>Giờ khám:</strong> ${this.formatTime(appointment.appointment_time)}</li>
      </ul>
      <p>Vui lòng đặt lại lịch nếu cần. Lưu ý: vui lòng đến sớm trước 15 phút để tránh huỷ lịch tự động.</p>
      <p>Trân trọng,<br>Phòng khám</p>
    `;

    return this.sendEmail({
      to: patientEmail,
      subject: 'Lịch hẹn bị huỷ',
      html,
    });
  }

  async sendAppointmentReminder(appointment, patientEmail) {
    const html = `
      <h2>Nhắc nhở lịch hẹn</h2>
      <p>Kính gửi bệnh nhân,</p>
      <p>Đây là email nhắc nhở lịch hẹn của bạn:</p>
      <ul>
        <li><strong>Bác sĩ:</strong> ${appointment.doctor.user.full_name}</li>
        <li><strong>Ngày khám:</strong> ${new Date(appointment.appointment_date).toLocaleDateString('vi-VN')}</li>
        <li><strong>Giờ khám:</strong> ${this.formatTime(appointment.appointment_time)}</li>
      </ul>
      <p>Vui lòng đến đúng giờ. Xin cảm ơn!</p>
      <p>Trân trọng,<br>Phòng khám</p>
    `;

    return this.sendEmail({
      to: patientEmail,
      subject: 'Nhắc nhở lịch hẹn khám bệnh',
      html,
    });
  }

  async sendPrescription(prescription, patientEmail, pdfPath) {
    const html = `
      <h2>Đơn thuốc điện tử</h2>
      <p>Kính gửi bệnh nhân,</p>
      <p>Đơn thuốc của bạn đã sẵn sàng. Vui lòng xem file đính kèm.</p>
      <p>Tổng tiền: ${prescription.total_amount.toLocaleString('vi-VN')} VNĐ</p>
      <p>Trân trọng,<br>Phòng khám</p>
    `;

    return this.sendEmail({
      to: patientEmail,
      subject: 'Đơn thuốc điện tử',
      html,
      attachments: pdfPath ? [{
        filename: 'prescription.pdf',
        path: pdfPath,
      }] : [],
    });
  }

  async sendLabResults(labOrder, patientEmail, pdfPath) {
    const html = `
      <h2>Kết quả xét nghiệm</h2>
      <p>Kính gửi bệnh nhân,</p>
      <p>Kết quả xét nghiệm của bạn đã có. Vui lòng xem file đính kèm.</p>
      <p>Nếu có thắc mắc, vui lòng liên hệ bác sĩ điều trị.</p>
      <p>Trân trọng,<br>Phòng khám</p>
    `;

    return this.sendEmail({
      to: patientEmail,
      subject: 'Kết quả xét nghiệm',
      html,
      attachments: pdfPath ? [{
        filename: 'lab-results.pdf',
        path: pdfPath,
      }] : [],
    });
  }
}

module.exports = new EmailService();