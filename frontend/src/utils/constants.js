// API Configuration
// Prefer CRA env var, fall back to Vite-style var if available at build time
export const API_BASE_URL = process.env.REACT_APP_API_URL || process.env.VITE_API_BASE_URL || 'http://localhost:3000/api';

// Roles
export const ROLES = {
  ADMIN: 'Admin',
  DOCTOR: 'Doctor',
  RECEPTIONIST: 'Receptionist',
  PHARMACIST: 'Pharmacist',
  LAB_TECH: 'LabTech',
  PATIENT: 'Patient',
};

// Appointment Status
export const APPOINTMENT_STATUS = {
  PENDING: 'PENDING',
  CONFIRMED: 'CONFIRMED',
  CHECKED_IN: 'CHECKED_IN',
  IN_PROGRESS: 'IN_PROGRESS',
  COMPLETED: 'COMPLETED',
  CANCELLED: 'CANCELLED',
  NO_SHOW: 'NO_SHOW',
};

export const APPOINTMENT_STATUS_LABELS = {
  PENDING: 'Chờ xác nhận',
  CONFIRMED: 'Đã xác nhận',
  CHECKED_IN: 'Đã check-in',
  IN_PROGRESS: 'Đang khám',
  COMPLETED: 'Hoàn thành',
  CANCELLED: 'Đã hủy',
  NO_SHOW: 'Không đến',
};

export const APPOINTMENT_STATUS_COLORS = {
  PENDING: 'warning',
  CONFIRMED: 'info',
  CHECKED_IN: 'primary',
  IN_PROGRESS: 'processing',
  COMPLETED: 'success',
  CANCELLED: 'error',
  NO_SHOW: 'default',
};

// Prescription Status
export const PRESCRIPTION_STATUS = {
  DRAFT: 'DRAFT',
  APPROVED: 'APPROVED',
  READY_FOR_DISPENSE: 'READY_FOR_DISPENSE',
  DISPENSED: 'DISPENSED',
};

export const PRESCRIPTION_STATUS_LABELS = {
  DRAFT: 'Bản nháp',
  APPROVED: 'Đã duyệt',
  READY_FOR_DISPENSE: 'Sẵn sàng cấp thuốc',
  DISPENSED: 'Đã phát thuốc',
};

// Invoice Status
export const INVOICE_STATUS = {
  UNPAID: 'UNPAID',
  PAID: 'PAID',
  REFUNDED: 'REFUNDED',
};

export const INVOICE_STATUS_LABELS = {
  UNPAID: 'Chưa thanh toán',
  PAID: 'Đã thanh toán',
  REFUNDED: 'Đã hoàn tiền',
};

// Lab Order Status
export const LAB_STATUS = {
  PENDING: 'PENDING',
  IN_PROGRESS: 'IN_PROGRESS',
  COMPLETED: 'COMPLETED',
};

export const LAB_STATUS_LABELS = {
  PENDING: 'Chờ xử lý',
  IN_PROGRESS: 'Đang xét nghiệm',
  COMPLETED: 'Hoàn thành',
};

// Gender
export const GENDER_OPTIONS = [
  { value: 'Male', label: 'Nam' },
  { value: 'Female', label: 'Nữ' },
  { value: 'Other', label: 'Khác' },
];

// Blood Types
export const BLOOD_TYPE_OPTIONS = [
  { value: 'A+', label: 'A+' },
  { value: 'A-', label: 'A-' },
  { value: 'B+', label: 'B+' },
  { value: 'B-', label: 'B-' },
  { value: 'AB+', label: 'AB+' },
  { value: 'AB-', label: 'AB-' },
  { value: 'O+', label: 'O+' },
  { value: 'O-', label: 'O-' },
];

// Specialties - 7 chuyên khoa chính
export const SPECIALTY_OPTIONS = [
  { value: 'Nội tổng quát', label: 'Nội tổng quát' },
  { value: 'Tim mạch', label: 'Tim mạch' },
  { value: 'Tiêu hóa', label: 'Tiêu hóa' },
  { value: 'Nội tiết', label: 'Nội tiết' },
  { value: 'Da liễu', label: 'Da liễu' },
  { value: 'Tai Mũi Họng', label: 'Tai Mũi Họng' },
  { value: 'Hô hấp', label: 'Hô hấp' },
];

// Pagination
export const PAGINATION = {
  DEFAULT_PAGE: 1,
  DEFAULT_LIMIT: 10,
  PAGE_SIZE_OPTIONS: [10, 20, 50, 100],
};

// Date Formats
export const DATE_FORMAT = 'DD/MM/YYYY';
export const TIME_FORMAT = 'HH:mm';
export const DATETIME_FORMAT = 'DD/MM/YYYY HH:mm';

// File Upload
export const FILE_UPLOAD = {
  MAX_SIZE: 5 * 1024 * 1024, // 5MB
  ALLOWED_IMAGE_TYPES: ['image/jpeg', 'image/png', 'image/jpg'],
  ALLOWED_DOCUMENT_TYPES: ['application/pdf', 'image/jpeg', 'image/png'],
};

// Notification Types
export const NOTIFICATION_TYPES = {
  APPOINTMENT_CREATED: 'APPOINTMENT_CREATED',
  APPOINTMENT_CONFIRMED: 'APPOINTMENT_CONFIRMED',
  APPOINTMENT_REMINDER: 'APPOINTMENT_REMINDER',
  APPOINTMENT_CANCELLED: 'APPOINTMENT_CANCELLED',
  PATIENT_CHECKED_IN: 'PATIENT_CHECKED_IN',
  PRESCRIPTION_READY: 'PRESCRIPTION_READY',
  LAB_RESULT_READY: 'LAB_RESULT_READY',
  LOW_STOCK: 'LOW_STOCK',
  EXPIRING_MEDICINES: 'EXPIRING_MEDICINES',
  HIGH_RISK_ALERT: 'HIGH_RISK_ALERT',
};

export const NOTIFICATION_TYPE_LABELS = {
  APPOINTMENT_CREATED: 'Lịch hẹn mới',
  APPOINTMENT_CONFIRMED: 'Xác nhận lịch hẹn',
  APPOINTMENT_REMINDER: 'Nhắc lịch hẹn',
  APPOINTMENT_CANCELLED: 'Hủy lịch hẹn',
  PATIENT_CHECKED_IN: 'Bệnh nhân check-in',
  PRESCRIPTION_READY: 'Đơn thuốc sẵn sàng',
  LAB_RESULT_READY: 'Kết quả xét nghiệm',
  LOW_STOCK: 'Thuốc sắp hết',
  EXPIRING_MEDICINES: 'Thuốc sắp hết hạn',
  HIGH_RISK_ALERT: 'Cảnh báo nguy cơ cao',
};