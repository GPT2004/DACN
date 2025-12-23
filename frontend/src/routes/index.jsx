/* eslint-disable no-unused-vars */
// frontend/src/routes/index.jsx
import { lazy } from 'react';
import { ROLES } from '../utils/constants';

// Lazy load pages
const LoginPage = lazy(() => import('../pages/auth/LoginPage'));
const RegisterPage = lazy(() => import('../pages/auth/RegisterPage'));
const ForgotPasswordPage = lazy(() => import('../pages/auth/ForgotPasswordPage'));
const ConfirmRegistrationPage = lazy(() => import('../pages/auth/ConfirmRegistration'));

// Patient Pages
const PatientDashboard = lazy(() => import('../pages/patient/PatientDashboard'));
const AppointmentsPage = lazy(() => import('../pages/patient/AppointmentsPage'));
const PatientAppointmentsPage = lazy(() => import('../pages/patient/PatientAppointmentsPage'));
const BookAppointmentPage = lazy(() => import('../pages/patient/BookAppointmentPage'));
const MedicalRecordsPage = lazy(() => import('../pages/patient/MedicalRecordsPage'));
const MessagesPage = lazy(() => import('../pages/patient/MessagesPage'));
const FAQPage = lazy(() => import('../pages/patient/FAQPage'));
const PrescriptionsPage = lazy(() => import('../pages/patient/PrescriptionsPage'));
const LabResultsPage = lazy(() => import('../pages/patient/LabResultsPage'));
const InvoicesPage = lazy(() => import('../pages/patient/InvoicesPage'));
const InvoicesDetailPage = lazy(() => import('../pages/patient/InvoicesDetailPage'));
const AISymptomCheckerPage = lazy(() => import('../pages/patient/AISymptomCheckerPage'));
const ProfilePage = lazy(() => import('../pages/patient/ProfilePage'));
const DoctorPublicDetail = lazy(() => import('../pages/patient/DoctorDetail'));

// Doctor Pages
const DoctorDashboard = lazy(() => import('../pages/doctor/DoctorDashboard'));
const SchedulePage = lazy(() => import('../pages/doctor/SchedulePage'));
const TodayAppointmentsPage = lazy(() => import('../pages/doctor/TodayAppointmentsPage'));
const ExaminationPage = lazy(() => import('../pages/doctor/ExaminationPage'));
const PatientDetailPage = lazy(() => import('../pages/doctor/PatientDetailPage'));
const PrescriptionPage = lazy(() => import('../pages/doctor/PrescriptionPage'));
const LabOrderPage = lazy(() => import('../pages/doctor/LabOrderPage'));
const AIToolsPage = lazy(() => import('../pages/doctor/AIToolsPage'));
const DoctorProfilePage = lazy(() => import('../pages/doctor/DoctorProfilePage'));
const DoctorMedicalRecordsPage = lazy(() => import('../pages/doctor/MedicalRecordsPage'));

// Receptionist Pages
const ReceptionistDashboard = lazy(() => import('../pages/receptionist/ReceptionistDashboardV2'));
const AppointmentManagementPage = lazy(() => import('../pages/receptionist/AppointmentManagementPage'));
const PatientRegistrationPage = lazy(() => import('../pages/receptionist/PatientRegistrationPage'));
const CheckInPage = lazy(() => import('../pages/receptionist/CheckInPage'));
const InvoicePage = lazy(() => import('../pages/receptionist/InvoicePage'));
const PrescriptionsInbox = lazy(() => import('../pages/receptionist/PrescriptionsInbox'));
const TimeslotManagementPage = lazy(() => import('../pages/receptionist/TimeslotManagementPage'));
const DailyReportPage = lazy(() => import('../pages/receptionist/DailyReportPage'));

// LabTech Pages
const LabTechDashboard = lazy(() => import('../pages/labtech/Dashboard'));
const LabOrdersPage = lazy(() => import('../pages/labtech/LabOrdersPage'));

// Pharmacist Pages
const PharmacistDashboard = lazy(() => import('../pages/pharmacist/Dashboard'));
const PharmacistMedicinesPage = lazy(() => import('../pages/pharmacist/MedicinesPage'));
const PharmacistPrescriptionsPage = lazy(() => import('../pages/pharmacist/PrescriptionsPage'));
const PharmacistInvoicesReadyPage = lazy(() => import('../pages/pharmacist/InvoicesReadyPage'));

// Admin Pages
const AdminDashboard = lazy(() => import('../pages/admin/AdminDashboard'));
const UsersPage = lazy(() => import('../pages/admin/UsersPage'));
const DoctorsPage = lazy(() => import('../pages/admin/DoctorsPage'));
const PatientsPage = lazy(() => import('../pages/admin/PatientsPage'));
const MedicinesPage = lazy(() => import('../pages/admin/MedicinesPage'));
const RoomsPage = lazy(() => import('../pages/admin/RoomsPage'));
const SchedulesPage = lazy(() => import('../pages/admin/SchedulesPage'));
const SuppliersPage = lazy(() => import('../pages/admin/SuppliersPage'));
const ReportsPage = lazy(() => import('../pages/admin/ReportsPage'));
const AuditLogsPage = lazy(() => import('../pages/admin/AuditLogsPage'));
const NotificationsPage = lazy(() => import('../pages/admin/NotificationsPage'));
const SettingsPage = lazy(() => import('../pages/admin/SettingsPage'));
const SpecialtyPage = lazy(() => import('../pages/admin/SpecialtyPage'));

// Home Page
const HomePage = lazy(() => import('../pages/HomePage'));
const SpecialtyDetail = lazy(() => import('../pages/SpecialtyDetail'));

// Error Pages
const NotFoundPage = lazy(() => import('../pages/NotFoundPage'));
const UnauthorizedPage = lazy(() => import('../pages/UnauthorizedPage'));

// Route configurations
export const publicRoutes = [
  {
    path: '/',
    element: HomePage,
    title: 'Trang chủ',
  },
  {
    path: '/specialty/:specialty',
    element: SpecialtyDetail,
    title: 'Chi tiết chuyên khoa',
  },
  {
    path: '/login',
    element: LoginPage,
    title: 'Đăng nhập',
  },
  {
    path: '/register',
    element: RegisterPage,
    title: 'Đăng ký',
  },
  {
    path: '/forgot-password',
    element: ForgotPasswordPage,
    title: 'Quên mật khẩu',
  },
  {
    path: '/confirm-registration',
    element: ConfirmRegistrationPage,
    title: 'Xác nhận đăng ký',
  },
  {
    path: '/doctors/:id',
    element: DoctorPublicDetail,
    title: 'Thông tin bác sĩ',
  },
];

export const patientRoutes = [
  {
    path: '/patient',
    element: PatientDashboard,
    title: 'Trang chủ',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/appointments',
    element: PatientAppointmentsPage,
    title: 'Lịch hẹn',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/appointments/book',
    element: BookAppointmentPage,
    title: 'Đặt lịch hẹn',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/medical-records',
    element: MedicalRecordsPage,
    title: 'Hồ sơ bệnh án',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/messages',
    element: MessagesPage,
    title: 'Tin nhắn',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/faq',
    element: FAQPage,
    title: 'Câu hỏi thường gặp',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/prescriptions',
    element: PrescriptionsPage,
    title: 'Đơn thuốc',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/lab-results',
    element: LabResultsPage,
    title: 'Kết quả xét nghiệm',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/invoices',
    element: InvoicesPage,
    title: 'Hóa đơn',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/invoices-detail',
    element: InvoicesDetailPage,
    title: 'Quản lý hóa đơn',
    role: ROLES.PATIENT,
  },
  // Appointment history page removed; history is presented inside the appointments page
  {
    path: '/patient/symptom-checker',
    element: AISymptomCheckerPage,
    title: 'Kiểm tra triệu chứng AI',
    role: ROLES.PATIENT,
  },
  {
    path: '/patient/profile',
    element: ProfilePage,
    title: 'Thông tin cá nhân',
    role: ROLES.PATIENT,
  },
];

export const doctorRoutes = [
  {
    path: '/doctor',
    element: DoctorDashboard,
    title: 'Trang chủ',
    role: ROLES.DOCTOR,
  },
  {
    path: '/doctor/schedule',
    element: SchedulePage,
    title: 'Lịch làm việc',
    role: ROLES.DOCTOR,
  },
  {
    path: '/doctor/appointments',
    element: TodayAppointmentsPage,
    title: 'Lịch hẹn khám bệnh',
    role: ROLES.DOCTOR,
  },
  {
    path: '/doctor/examination/:id',
    element: ExaminationPage,
    title: 'Khám bệnh',
    role: ROLES.DOCTOR,
  },
  {
    path: '/doctor/patients/:id',
    element: PatientDetailPage,
    title: 'Chi tiết bệnh nhân',
    role: ROLES.DOCTOR,
  },
  {
    path: '/doctor/prescriptions',
    element: PrescriptionPage,
    title: 'Kê đơn thuốc',
    role: ROLES.DOCTOR,
  },
  {
    path: '/doctor/lab-orders',
    element: LabOrderPage,
    title: 'Chỉ định xét nghiệm',
    role: ROLES.DOCTOR,
  },
  {
    path: '/doctor/ai-tools',
    element: AIToolsPage,
    title: 'Công cụ AI',
    role: ROLES.DOCTOR,
  },
  {
    path: '/doctor/profile',
    element: DoctorProfilePage,
    title: 'Thông tin cá nhân',
    role: ROLES.DOCTOR,
  },
  {
    path: '/doctor/medical-records',
    element: DoctorMedicalRecordsPage,
    title: 'Hồ sơ bệnh án',
    role: ROLES.DOCTOR,
  },
];

export const receptionistRoutes = [
  {
    path: '/receptionist',
    element: ReceptionistDashboard,
    title: 'Trang chủ',
    role: ROLES.RECEPTIONIST,
  },
  {
    path: '/receptionist/appointments',
    element: AppointmentManagementPage,
    title: 'Quản lý lịch hẹn',
    role: ROLES.RECEPTIONIST,
  },
  {
    path: '/receptionist/register-patient',
    element: PatientRegistrationPage,
    title: 'Đăng ký bệnh nhân',
    role: ROLES.RECEPTIONIST,
  },
  {
    path: '/receptionist/check-in',
    element: CheckInPage,
    title: 'Check-in bệnh nhân',
    role: ROLES.RECEPTIONIST,
  },
  {
    path: '/receptionist/invoices',
    element: InvoicePage,
    title: 'Quản lý hóa đơn',
    role: ROLES.RECEPTIONIST,
  },
  {
    path: '/receptionist/prescriptions-inbox',
    element: PrescriptionsInbox,
    title: 'Đơn thuốc chờ lập hoá đơn',
    role: ROLES.RECEPTIONIST,
  },
  {
    path: '/receptionist/timeslots',
    element: TimeslotManagementPage,
    title: 'Quản lý khung giờ',
    role: ROLES.RECEPTIONIST,
  },
  {
    path: '/receptionist/daily-report',
    element: DailyReportPage,
    title: 'Báo cáo hàng ngày',
    role: ROLES.RECEPTIONIST,
  },
];

export const labTechRoutes = [
  {
    path: '/labtech',
    element: LabTechDashboard,
    title: 'Trang chủ',
    role: ROLES.LAB_TECH,
  },
  {
    path: '/labtech/lab-orders',
    element: LabOrdersPage,
    title: 'Quản lý chỉ định xét nghiệm',
    role: ROLES.LAB_TECH,
  },
];

export const pharmacistRoutes = [
  {
    path: '/pharmacist',
    element: PharmacistDashboard,
    title: 'Trang chủ',
    role: ROLES.PHARMACIST,
  },
  {
    path: '/pharmacist/medicines',
    element: PharmacistMedicinesPage,
    title: 'Quản lý thuốc',
    role: ROLES.PHARMACIST,
  },
  {
    path: '/pharmacist/prescriptions',
    element: PharmacistPrescriptionsPage,
    title: 'Quản lý đơn thuốc',
    role: ROLES.PHARMACIST,
  },
  {
    path: '/pharmacist/invoices-ready',
    element: PharmacistInvoicesReadyPage,
    title: 'Hóa đơn sẵn sàng cấp thuốc',
    role: ROLES.PHARMACIST,
  },
];

export const adminRoutes = [
  {
    path: '/admin',
    element: AdminDashboard,
    title: 'Trang quản trị',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/users',
    element: UsersPage,
    title: 'Quản lý người dùng',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/doctors',
    element: DoctorsPage,
    title: 'Quản lý bác sĩ',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/patients',
    element: PatientsPage,
    title: 'Quản lý bệnh nhân',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/medicines',
    element: MedicinesPage,
    title: 'Quản lý thuốc',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/rooms',
    element: RoomsPage,
    title: 'Quản lý phòng khám',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/specialties',
    element: SpecialtyPage,
    title: 'Quản lý chuyên khoa',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/schedules',
    element: SchedulesPage,
    title: 'Quản lý lịch làm việc',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/suppliers',
    element: SuppliersPage,
    title: 'Quản lý nhà cung cấp',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/reports',
    element: ReportsPage,
    title: 'Báo cáo & Thống kê',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/audit-logs',
    element: AuditLogsPage,
    title: 'Nhật ký hệ thống',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/notifications',
    element: NotificationsPage,
    title: 'Quản lý thông báo',
    role: ROLES.ADMIN,
  },
  {
    path: '/admin/settings',
    element: SettingsPage,
    title: 'Cài đặt hệ thống',
    role: ROLES.ADMIN,
  },
];

export const errorRoutes = [
  {
    path: '/unauthorized',
    element: UnauthorizedPage,
    title: 'Không có quyền truy cập',
  },
  {
    path: '*',
    element: NotFoundPage,
    title: 'Không tìm thấy trang',
  },
];

// All protected routes
export const protectedRoutes = [
  ...patientRoutes,
  ...doctorRoutes,
  ...receptionistRoutes,
  ...labTechRoutes,
  ...pharmacistRoutes,
  ...adminRoutes,
];

// Get routes by role
export const getRoutesByRole = (role) => {
  switch (role) {
    case ROLES.PATIENT:
      return patientRoutes;
    case ROLES.DOCTOR:
      return doctorRoutes;
    case ROLES.RECEPTIONIST:
      return receptionistRoutes;
    case ROLES.LAB_TECH:
      return labTechRoutes;
    case ROLES.PHARMACIST:
      return pharmacistRoutes;
    case ROLES.ADMIN:
      return adminRoutes;
    default:
      return [];
  }
};

// Get default path by role
export const getDefaultPathByRole = (role) => {
  switch (role) {
    case ROLES.PATIENT:
      return '/patient';
    case ROLES.DOCTOR:
      return '/doctor';
    case ROLES.RECEPTIONIST:
      return '/receptionist';
    case ROLES.LAB_TECH:
      return '/labtech';
    case ROLES.PHARMACIST:
      return '/pharmacist';
    case ROLES.ADMIN:
      return '/admin';
    default:
      return '/login';
  }
};