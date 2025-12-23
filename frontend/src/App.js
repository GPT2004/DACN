import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';

// Home Page
import HomePage from './pages/HomePage';
import SpecialtyDetail from './pages/SpecialtyDetail';

// Auth Pages
import LoginPage from './pages/auth/LoginPage';
import RegisterPage from './pages/auth/RegisterPage';
import ForgotPasswordPage from './pages/auth/ForgotPasswordPage';

// Patient Pages
import PatientDashboard from './pages/patient/PatientDashboard';
// eslint-disable-next-line no-unused-vars
import AppointmentsPage from './pages/patient/AppointmentsPage';
import BookAppointmentPage from './pages/patient/BookAppointmentPage';
import PatientAppointmentsPage from './pages/patient/PatientAppointmentsPage';
import AppointmentHistoryPage from './pages/patient/AppointmentHistoryPage';
import InvoicesDetailPage from './pages/patient/InvoicesDetailPage';
import MedicalRecordsPage from './pages/patient/MedicalRecordsPage';
import PrescriptionsPage from './pages/patient/PrescriptionsPage';
import AISymptomCheckerPage from './pages/patient/AISymptomCheckerPage';
import ProfilePage from './pages/patient/ProfilePage';
import MessagesPage from './pages/patient/MessagesPage';
import FAQPage from './pages/patient/FAQPage';
import DoctorDetail from './pages/patient/DoctorDetail';

// Doctor Pages
import DoctorDashboard from './pages/doctor/DoctorDashboard';
import SchedulePage from './pages/doctor/SchedulePage';
import TodayAppointmentsPage from './pages/doctor/TodayAppointmentsPage';
import ExaminationPage from './pages/doctor/ExaminationPage';
import AIToolsPage from './pages/doctor/AIToolsPage';
import DoctorProfilePage from './pages/doctor/DoctorProfilePage';
import PrescriptionPage from './pages/doctor/PrescriptionPage';
import DoctorMedicalRecordsPage from './pages/doctor/MedicalRecordsPage';
import DoctorLayout from './components/doctor/DoctorLayout';

// Receptionist Pages
// eslint-disable-next-line no-unused-vars
import ReceptionistDashboard from './pages/receptionist/ReceptionistDashboard';
import ReceptionistDashboardV2 from './pages/receptionist/ReceptionistDashboardV2';
import AppointmentManagementPage from './pages/receptionist/AppointmentManagementPage';
import PatientRegistrationPage from './pages/receptionist/PatientRegistrationPage';
import CheckInPage from './pages/receptionist/CheckInPage';
import InvoicePage from './pages/receptionist/InvoicePage';
import TimeslotManagementPage from './pages/receptionist/TimeslotManagementPage';
import DailyReportPage from './pages/receptionist/DailyReportPage';

// Pharmacist Pages
import PharmacistDashboard from './pages/pharmacist/Dashboard';
import PharmacistPrescriptionsPage from './pages/pharmacist/PrescriptionsPage';
import PharmacistMedicinesPage from './pages/pharmacist/MedicinesPage';
import PharmacistInvoicesReadyPage from './pages/pharmacist/InvoicesReadyPage';

// LabTech Pages
import LabTechDashboard from './pages/labtech/Dashboard';
import LabTechLabOrdersPage from './pages/labtech/LabOrdersPage';

// Admin Pages
import AdminDashboard from './pages/admin/AdminDashboard';
import UsersPage from './pages/admin/UsersPage';
import DoctorsPage from './pages/admin/DoctorsPage';
// eslint-disable-next-line no-unused-vars
import MedicinesPage from './pages/admin/MedicinesPage';
import ReportsPage from './pages/admin/ReportsPage';
import AuditLogsPage from './pages/admin/AuditLogsPage';
import RecycleBinPage from './pages/admin/RecycleBinPage';
import RoomsPage from './pages/admin/RoomsPage';
import SpecialtyPage from './pages/admin/SpecialtyPage';

// Error Pages
import NotFoundPage from './pages/NotFoundPage';
import UnauthorizedPage from './pages/UnauthorizedPage';

// Protected Route
import ProtectedRoute from './components/auth/ProtectedRoute';
import DashboardLayout from './components/layout/DashboardLayout';

function App() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  // ✅ FIX LỖI ĐOẠN NÀY: map role strings to their dashboard paths
  const userRole = (user?.role?.name || user?.role || "").toString();
  const roleMap = {
    admin: "/admin",
    doctor: "/doctor",
    patient: "/patient",
    receptionist: "/receptionist",
    reception: "/receptionist",
    pharmacist: "/pharmacist",
    labtech: "/labtech",
  };
  const redirectPath = roleMap[userRole.toLowerCase()] || "/";

  return (
    <Router>
      <Routes>

        {/* Home Page (Public) */}
        <Route path="/" element={<HomePage />} />

        {/* Specialty Detail (Public) */}
        <Route path="/specialty/:specialty" element={<SpecialtyDetail />} />

        {/* Public */}
        <Route
          path="/login"
          element={!user ? <LoginPage /> : <Navigate to={redirectPath} replace />}
        />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />

        {/* Public Doctor Detail */}
        <Route path="/doctors/:id" element={<DoctorDetail />} />

        {/* Patient */}
        <Route path="/patient" element={<ProtectedRoute role="patient"><DashboardLayout role="patient" showSidebar={false}><PatientDashboard /></DashboardLayout></ProtectedRoute>} />
        <Route path="/patient/appointments" element={<ProtectedRoute role="patient"><PatientAppointmentsPage /></ProtectedRoute>} />
        <Route path="/patient/appointment-history" element={<ProtectedRoute role="patient"><AppointmentHistoryPage /></ProtectedRoute>} />
        <Route path="/patient/appointments/book" element={<ProtectedRoute role="patient"><BookAppointmentPage /></ProtectedRoute>} />
        <Route path="/patient/medical-records" element={<ProtectedRoute role="patient"><MedicalRecordsPage /></ProtectedRoute>} />
        <Route path="/patient/invoices-detail" element={<ProtectedRoute role="patient"><InvoicesDetailPage /></ProtectedRoute>} />
        <Route path="/patient/prescriptions" element={<ProtectedRoute role="patient"><DashboardLayout role="patient" showSidebar={false}><PrescriptionsPage /></DashboardLayout></ProtectedRoute>} />
        <Route path="/patient/symptom-checker" element={<ProtectedRoute role="patient"><DashboardLayout role="patient" showSidebar={false}><AISymptomCheckerPage /></DashboardLayout></ProtectedRoute>} />
        <Route path="/patient/profile" element={<ProtectedRoute role="patient"><ProfilePage /></ProtectedRoute>} />
        <Route path="/patient/messages" element={<ProtectedRoute role="patient"><MessagesPage /></ProtectedRoute>} />
        <Route path="/patient/faq" element={<ProtectedRoute role="patient"><FAQPage /></ProtectedRoute>} />

        {/* Doctor (these pages provide their own admin-style layout) */}
        <Route path="/doctor" element={<ProtectedRoute role="doctor"><DoctorDashboard /></ProtectedRoute>} />
        <Route path="/doctor/schedule" element={<ProtectedRoute role="doctor"><SchedulePage /></ProtectedRoute>} />
        <Route path="/doctor/appointments" element={<ProtectedRoute role="doctor"><TodayAppointmentsPage /></ProtectedRoute>} />
        <Route path="/doctor/profile" element={<ProtectedRoute role="doctor"><DoctorProfilePage /></ProtectedRoute>} />
        <Route path="/doctor/prescriptions" element={<ProtectedRoute role="doctor"><DoctorLayout><PrescriptionPage /></DoctorLayout></ProtectedRoute>} />
        <Route path="/doctor/medical-records" element={<ProtectedRoute role="doctor"><DoctorLayout><DoctorMedicalRecordsPage /></DoctorLayout></ProtectedRoute>} />
        <Route path="/doctor/examination/:id" element={<ProtectedRoute role="doctor"><ExaminationPage /></ProtectedRoute>} />
        <Route path="/doctor/ai-tools" element={<ProtectedRoute role="doctor"><AIToolsPage /></ProtectedRoute>} />

        {/* Receptionist (pages provide their own layout) */}
        <Route path="/receptionist" element={<ProtectedRoute role="receptionist"><ReceptionistDashboardV2 /></ProtectedRoute>} />
        <Route path="/receptionist/appointments" element={<ProtectedRoute role="receptionist"><AppointmentManagementPage /></ProtectedRoute>} />
        <Route path="/receptionist/register-patient" element={<ProtectedRoute role="receptionist"><PatientRegistrationPage /></ProtectedRoute>} />
        <Route path="/receptionist/check-in" element={<ProtectedRoute role="receptionist"><CheckInPage /></ProtectedRoute>} />
        <Route path="/receptionist/invoices" element={<ProtectedRoute role="receptionist"><InvoicePage /></ProtectedRoute>} />
        <Route path="/receptionist/timeslots" element={<ProtectedRoute role="receptionist"><TimeslotManagementPage /></ProtectedRoute>} />
        <Route path="/receptionist/daily-report" element={<ProtectedRoute role="receptionist"><DailyReportPage /></ProtectedRoute>} />

        {/* Pharmacist (pages provide their own layout) */}
        <Route path="/pharmacist" element={<ProtectedRoute role="pharmacist"><PharmacistDashboard /></ProtectedRoute>} />
        <Route path="/pharmacist/prescriptions" element={<ProtectedRoute role="pharmacist"><PharmacistPrescriptionsPage /></ProtectedRoute>} />
        <Route path="/pharmacist/medicines" element={<ProtectedRoute role="pharmacist"><PharmacistMedicinesPage /></ProtectedRoute>} />
        <Route path="/pharmacist/invoices-ready" element={<ProtectedRoute role="pharmacist"><PharmacistInvoicesReadyPage /></ProtectedRoute>} />

        {/* LabTech (pages provide their own layout) */}
        <Route path="/labtech" element={<ProtectedRoute role="labtech"><LabTechDashboard /></ProtectedRoute>} />
        <Route path="/labtech/lab-orders" element={<ProtectedRoute role="labtech"><LabTechLabOrdersPage /></ProtectedRoute>} />

        {/* Admin (pages provide their own layout) */}
        <Route path="/admin" element={<ProtectedRoute role="admin"><AdminDashboard /></ProtectedRoute>} />
        <Route path="/admin/users" element={<ProtectedRoute role="admin"><UsersPage /></ProtectedRoute>} />
        <Route path="/admin/doctors" element={<ProtectedRoute role="admin"><DoctorsPage /></ProtectedRoute>} />
        <Route path="/admin/rooms" element={<ProtectedRoute role="admin"><RoomsPage /></ProtectedRoute>} />
        <Route path="/admin/specialties" element={<ProtectedRoute role="admin"><SpecialtyPage /></ProtectedRoute>} />
        <Route path="/admin/audit-logs" element={<ProtectedRoute role="admin"><AuditLogsPage /></ProtectedRoute>} />
        <Route path="/admin/recycle-bin" element={<ProtectedRoute role="admin"><RecycleBinPage /></ProtectedRoute>} />
        {/* Medicines moved to pharmacist role; removed admin medicines route */}
        <Route path="/admin/reports" element={<ProtectedRoute role="admin"><ReportsPage /></ProtectedRoute>} />

        {/* Default redirect */}
        <Route path="/*" element={<Navigate to={user ? redirectPath : "/"} replace />} />

        {/* Errors */}
        <Route path="/unauthorized" element={<UnauthorizedPage />} />
        <Route path="*" element={<NotFoundPage />} />

      </Routes>
    </Router>
  );
}

export default App;
