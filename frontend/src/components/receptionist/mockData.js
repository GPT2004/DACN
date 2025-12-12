// Mock data for receptionist components

export const mockAppointments = [
  {
    id: 1,
    patientId: 'BN001',
    patientName: 'Nguyễn Văn A',
    doctorName: 'TS. Trần B',
    time: '08:00',
    date: '2025-11-17',
    status: 'CONFIRMED',
    reason: 'Khám tổng quát'
  },
  {
    id: 2,
    patientId: 'BN002',
    patientName: 'Phạm Thị C',
    doctorName: 'TS. Nguyễn D',
    time: '08:30',
    date: '2025-11-17',
    status: 'CONFIRMED',
    reason: 'Khám nhi'
  },
  {
    id: 3,
    patientId: 'BN003',
    patientName: 'Trần Văn E',
    doctorName: 'TS. Lê F',
    time: '09:00',
    date: '2025-11-17',
    status: 'PENDING',
    reason: 'Tái khám'
  },
  {
    id: 4,
    patientId: 'BN004',
    patientName: 'Lý Thị G',
    doctorName: 'TS. Trần B',
    time: '09:30',
    date: '2025-11-17',
    status: 'CONFIRMED',
    reason: 'Khám ngoài'
  },
  {
    id: 5,
    patientId: 'BN005',
    patientName: 'Hoàng Văn H',
    doctorName: 'TS. Nguyễn D',
    time: '10:00',
    date: '2025-11-17',
    status: 'CONFIRMED',
    reason: 'Khám tổng quát'
  },
  {
    id: 6,
    patientId: 'BN006',
    patientName: 'Võ Thị I',
    doctorName: 'TS. Lê F',
    time: '10:30',
    date: '2025-11-17',
    status: 'CHECKED_IN',
    reason: 'Xét nghiệm'
  },
  {
    id: 7,
    patientId: 'BN007',
    patientName: 'Bùi Văn K',
    doctorName: 'TS. Trần B',
    time: '13:00',
    date: '2025-11-17',
    status: 'CONFIRMED',
    reason: 'Tái khám'
  },
  {
    id: 8,
    patientId: 'BN008',
    patientName: 'Đỗ Thị L',
    doctorName: 'TS. Nguyễn D',
    time: '13:30',
    date: '2025-11-17',
    status: 'CONFIRMED',
    reason: 'Khám ngoài'
  },
  {
    id: 9,
    patientId: 'BN009',
    patientName: 'Dương Văn M',
    doctorName: 'TS. Lê F',
    time: '14:00',
    date: '2025-11-17',
    status: 'PENDING',
    reason: 'Khám tổng quát'
  },
  {
    id: 10,
    patientId: 'BN010',
    patientName: 'Chu Thị N',
    doctorName: 'TS. Trần B',
    time: '14:30',
    date: '2025-11-17',
    status: 'CONFIRMED',
    reason: 'Khám nhi'
  }
];

export const mockTimeSlots = [
  { id: 1, time: '08:00-08:30', maxPatients: 5 },
  { id: 2, time: '08:30-09:00', maxPatients: 5 },
  { id: 3, time: '09:00-09:30', maxPatients: 5 },
  { id: 4, time: '09:30-10:00', maxPatients: 5 },
  { id: 5, time: '10:00-10:30', maxPatients: 5 },
  { id: 6, time: '10:30-11:00', maxPatients: 5 },
  { id: 7, time: '13:00-13:30', maxPatients: 5 },
  { id: 8, time: '13:30-14:00', maxPatients: 5 },
  { id: 9, time: '14:00-14:30', maxPatients: 5 },
  { id: 10, time: '14:30-15:00', maxPatients: 5 },
  { id: 11, time: '15:00-15:30', maxPatients: 5 },
  { id: 12, time: '15:30-16:00', maxPatients: 5 }
];

export const mockPatients = [
  {
    id: 1,
    user: { full_name: 'Nguyễn Văn A', phone: '0901234567', email: 'nguyenvana@email.com' },
    gender: 'Nam',
    dob: '1990-01-15',
    address: '123 Đường A, Quận 1, TP.HCM'
  },
  {
    id: 2,
    user: { full_name: 'Phạm Thị C', phone: '0912345678', email: 'phamthic@email.com' },
    gender: 'Nữ',
    dob: '1995-05-20',
    address: '456 Đường B, Quận 2, TP.HCM'
  },
  {
    id: 3,
    user: { full_name: 'Trần Văn E', phone: '0923456789', email: 'tranvane@email.com' },
    gender: 'Nam',
    dob: '1988-09-10',
    address: '789 Đường C, Quận 3, TP.HCM'
  },
  {
    id: 4,
    user: { full_name: 'Lý Thị G', phone: '0934567890', email: 'lythig@email.com' },
    gender: 'Nữ',
    dob: '2000-03-25',
    address: '012 Đường D, Quận 4, TP.HCM'
  },
  {
    id: 5,
    user: { full_name: 'Hoàng Văn H', phone: '0945678901', email: 'hoangvanh@email.com' },
    gender: 'Nam',
    dob: '1992-11-08',
    address: '345 Đường E, Quận 5, TP.HCM'
  }
];

export const mockDoctors = [
  {
    id: 1,
    user: { full_name: 'TS. Trần B', phone: '0956789012', email: 'tranb@clinic.com' },
    specialization: 'Nhi',
    rating: 4.8
  },
  {
    id: 2,
    user: { full_name: 'TS. Nguyễn D', phone: '0967890123', email: 'nguyend@clinic.com' },
    specialization: 'Ngoài',
    rating: 4.6
  },
  {
    id: 3,
    user: { full_name: 'TS. Lê F', phone: '0978901234', email: 'lef@clinic.com' },
    specialization: 'Tổng quát',
    rating: 4.7
  }
];

export const mockInvoices = [
  {
    id: 1,
    invoice_no: 'INV-001',
    patientName: 'Nguyễn Văn A',
    amount: 500000,
    status: 'UNPAID'
  },
  {
    id: 2,
    invoice_no: 'INV-002',
    patientName: 'Phạm Thị C',
    amount: 750000,
    status: 'UNPAID'
  },
  {
    id: 3,
    invoice_no: 'INV-003',
    patientName: 'Trần Văn E',
    amount: 600000,
    status: 'PAID'
  }
];
