import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import ReceptionistLayout from '../../components/layout/ReceptionistLayout';
import DetailModal from '../../components/receptionist/DetailModal';
import AppointmentDetailModal from '../../components/receptionist/AppointmentDetailModal';
import PaymentModal from '../../components/receptionist/PaymentModal';
import CheckInModal from '../../components/receptionist/CheckInModal';
import ScheduleModal from '../../components/receptionist/ScheduleModal';
import { 
  Users, Calendar, DollarSign, CheckCircle2, 
  Plus, Edit2, Trash2, Eye,
  Phone, Mail, MapPin, TrendingUp,
  User, FileText, Stethoscope
} from 'lucide-react';
import { appointmentService } from '../../services/appointmentService';
import BookAppointmentForm from '../../components/patient/BookAppointmentForm';
import { patientService } from '../../services/patientService';
import PatientRegistrationForm from '../../components/receptionist/PatientRegistrationForm';
import { invoiceService } from '../../services/invoiceService';
import { prescriptionService } from '../../services/prescriptionService';
import { doctorService } from '../../services/doctorService';
import ReceptionistSidebar from '../../components/receptionist/ReceptionistSidebar';
import CreateInvoiceFromPrescriptionModal from '../../components/receptionist/CreateInvoiceFromPrescriptionModal';

export default function ExpandedReceptionistDashboard() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [appointments, setAppointments] = useState([]);
  const [patients, setPatients] = useState([]);
  const [invoices, setInvoices] = useState([]);
  const [pendingPrescriptions, setPendingPrescriptions] = useState([]);
  const [doctors, setDoctors] = useState([]);
  const [showPrescriptionDropdown, setShowPrescriptionDropdown] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [selectedItemType, setSelectedItemType] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [scheduleDoctor, setScheduleDoctor] = useState(null);
  const [showPaymentModal, setShowPaymentModal] = useState(false);
  const [showCheckInModal, setShowCheckInModal] = useState(false);
  const [showAddAppointmentModal, setShowAddAppointmentModal] = useState(false);
  const [showAddPatientModal, setShowAddPatientModal] = useState(false);
  const [showCreateFromPrescriptionModal, setShowCreateFromPrescriptionModal] = useState(false);
  const [modalPrescriptionId, setModalPrescriptionId] = useState(null);
  const [creatingPatient, setCreatingPatient] = useState(false);
  const [patientError, setPatientError] = useState('');
  // when creating quick patients we do not generate/store passwords
  const [appointmentInitialValues, setAppointmentInitialValues] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [stats, setStats] = useState({
    totalPatients: 0,
    todayAppointments: 0,
    pendingInvoices: 0,
    checkedInToday: 0
  });

  // Fetch all data
  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      const today = (() => {
        const d = new Date();
        const y = d.getFullYear();
        const m = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        return `${y}-${m}-${day}`;
      })();

      // Fetch appointments
      const aptsRes = await appointmentService.getAppointments({ limit: 100, page: 1 })
        .catch(() => ({ data: [] }));
      // DEBUG: log raw appointments response to help diagnose missing items
      // (Remove or disable in production)
      console.debug('DEBUG appointments response:', aptsRes);
      const aptList = Array.isArray(aptsRes?.data) ? aptsRes.data : (aptsRes?.data?.appointments || []);
      
      console.debug('DEBUG parsed appointment list:', aptList);
      const todayApts = aptList.filter(apt => {
        const dt = parseAppointmentDateTime(apt) || new Date(apt.appointment_date || Date.now());
        const y = dt.getFullYear();
        const m = String(dt.getMonth() + 1).padStart(2, '0');
        const day = String(dt.getDate()).padStart(2, '0');
        const aptDate = `${y}-${m}-${day}`;
        return aptDate === today;
      });
      // Show all appointments in the table (not only today's)
      setAppointments(aptList);

      // Fetch patients from database
      const ptsRes = await patientService.getPatients({ limit: 100, page: 1 })
        .catch(() => ({ data: { patients: [] } }));
      let ptList = ptsRes?.data?.patients || [];
      setPatients(ptList);

      // Fetch invoices from database
      const invRes = await invoiceService.getInvoices({ limit: 100, page: 1 })
        .catch(() => ({ data: { invoices: [] } }));
      let invList = invRes?.data?.invoices || [];
      const pendingInvs = invList.filter(inv => inv.status === 'UNPAID');
      setInvoices(invList);

      // Fetch prescriptions waiting for invoicing (for dashboard quick view)
      const presRes = await prescriptionService.getForInvoicing({ page: 1, limit: 10 }).catch(() => ({ data: { prescriptions: [] } }));
      const presList = presRes?.data?.prescriptions || presRes?.data || [];
      setPendingPrescriptions(presList);

      // Fetch doctors from database
      const docsRes = await doctorService.getDoctors({ limit: 50, page: 1 })
        .catch(() => ({ data: { doctors: [] } }));
      let docList = docsRes?.data?.doctors || [];
      setDoctors(docList);

      // Update stats
      const uniquePatients = new Set(aptList.map(apt => apt.patient_id)).size;
      const checkedIn = todayApts.filter(apt => apt.status === 'CHECKED_IN' || apt.status === 'IN_PROGRESS').length;

      setStats({
        totalPatients: uniquePatients || ptList.length,
        todayAppointments: todayApts.length,
        pendingInvoices: pendingInvs.length,
        checkedInToday: checkedIn
      });
    } catch (error) {
      // Handle error silently
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      'PENDING': 'bg-yellow-100 text-yellow-800',
      'CONFIRMED': 'bg-blue-100 text-blue-800',
      'CHECKED_IN': 'bg-purple-100 text-purple-800',
      'IN_PROGRESS': 'bg-indigo-100 text-indigo-800',
      'COMPLETED': 'bg-green-100 text-green-800',
      'CANCELLED': 'bg-red-100 text-red-800',
      'PAID': 'bg-green-100 text-green-800',
      'UNPAID': 'bg-orange-100 text-orange-800'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
  };

  const getStatusLabel = (status) => {
    const labels = {
      'PENDING': 'Chờ xác nhận',
      'CONFIRMED': 'Đã xác nhận',
      'CHECKED_IN': 'Đã check-in',
      'IN_PROGRESS': 'Đang khám',
      'COMPLETED': 'Hoàn thành',
      'CANCELLED': 'Đã hủy',
      'PAID': 'Đã thanh toán',
      'UNPAID': 'Chưa thanh toán'
    };
    return labels[status] || status;
  };

  // Parse appointment datetime robustly by combining date + time when time is a time-only string
  const parseAppointmentDateTime = (apt) => {
    try {
      // Prefer backend-provided combined ISO datetime if available
      if (apt?.appointment_datetime) {
        const dt = new Date(apt.appointment_datetime);
        if (!isNaN(dt.getTime())) return dt;
      }

      const dateStr = apt?.appointment_date || (apt?.timeslot && apt.timeslot.date);
      const timeRaw = apt?.appointment_time || (apt?.timeslot && apt.timeslot.start_time);

      let dt = null;

      if (dateStr && timeRaw) {
        // Parse date from "YYYY-MM-DD" string (NOT as UTC)
        let year = 2025, month = 1, day = 1;
        if (typeof dateStr === 'string') {
          const dateParts = dateStr.split('T')[0].split('-'); // Extract YYYY-MM-DD part
          if (dateParts.length >= 3) {
            year = parseInt(dateParts[0], 10);
            month = parseInt(dateParts[1], 10) - 1; // JS month is 0-indexed
            day = parseInt(dateParts[2], 10);
          }
        } else {
          const d = new Date(dateStr);
          year = d.getFullYear();
          month = d.getMonth();
          day = d.getDate();
        }

        // Parse time from "HH:MM:SS" string
        let hours = 0, minutes = 0, seconds = 0;
        if (typeof timeRaw === 'string' && /^\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(timeRaw)) {
          const timePart = timeRaw.split(':');
          hours = parseInt(timePart[0], 10);
          minutes = parseInt(timePart[1], 10);
          seconds = timePart[2] ? parseInt(timePart[2], 10) : 0;
        } else if (typeof timeRaw === 'string' && /^\d{4}-\d{2}-\d{2}T/.test(timeRaw)) {
          const t = new Date(timeRaw);
          if (!isNaN(t.getTime())) {
            hours = t.getHours();
            minutes = t.getMinutes();
            seconds = t.getSeconds();
          }
        }

        // Create as local time
        dt = new Date(year, month, day, hours, minutes, seconds);
      } else if (timeRaw) {
        dt = new Date(timeRaw);
      } else if (dateStr) {
        dt = new Date(dateStr);
      }

      if (dt && !isNaN(dt.getTime())) return dt;
    } catch (e) {
      // fallthrough
    }
    return null;
  };

  // Filtered data based on search
  const filteredAppointments = appointments.filter(apt =>
    apt.patient?.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    apt.doctor?.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const filteredPatients = patients.filter(pt =>
    pt.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    pt.user?.email?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const filteredInvoices = invoices.filter(inv =>
    inv.patient?.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    inv.invoice_number?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const filteredDoctors = doctors.filter(doc =>
    doc.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.specialization?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Tab Contents
  const tabs = [
    {
      id: 'overview',
      label: 'Tổng quan',
      icon: TrendingUp,
      content: (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6 shadow">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-700 text-sm font-medium">Tổng bệnh nhân</p>
                  <p className="text-3xl font-bold text-blue-900 mt-2">{stats.totalPatients}</p>
                </div>
                <Users className="text-blue-400" size={48} />
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6 shadow">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-purple-700 text-sm font-medium">Lịch hẹn khám bệnh</p>
                  <p className="text-3xl font-bold text-purple-900 mt-2">{stats.todayAppointments}</p>
                </div>
                <Calendar className="text-purple-400" size={48} />
              </div>
            </div>

            <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg p-6 shadow">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-orange-700 text-sm font-medium">Chờ thanh toán</p>
                  <p className="text-3xl font-bold text-orange-900 mt-2">{stats.pendingInvoices}</p>
                </div>
                <DollarSign className="text-orange-400" size={48} />
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6 shadow">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-green-700 text-sm font-medium">Đã check-in</p>
                  <p className="text-3xl font-bold text-green-900 mt-2">{stats.checkedInToday}</p>
                </div>
                <CheckCircle2 className="text-green-400" size={48} />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Calendar size={20} /> Lịch hẹn sắp tới
              </h3>
              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {appointments.slice(0, 10).map(apt => (
                  <div key={apt.id} className="p-3 bg-gray-50 rounded-lg border-l-4 border-purple-500 hover:bg-gray-100 cursor-pointer transition"
                    onClick={async () => {
                      try {
                        // Fetch full appointment data with patient and doctor details
                        const fullApt = await appointmentService.getAppointmentById(apt.id);
                        // Normalize response to the appointment object
                        const serverPayload = fullApt?.data ?? fullApt;
                        const appointmentObj = serverPayload?.data ?? serverPayload;

                        // Debug
                        console.log('📥 Full appointment fetched (raw):', fullApt);
                        console.log('📥 Normalized appointment object:', appointmentObj);

                        // Set state and wait for next tick before showing modal
                        setSelectedItem(appointmentObj || apt);
                        setSelectedItemType('appointment');

                        // Use setTimeout to ensure state is updated before modal renders
                        setTimeout(() => {
                          setShowDetailModal(true);
                        }, 50);
                      } catch (err) {
                        console.error('Error fetching appointment:', err);
                        // Fallback to basic data
                        setSelectedItem(apt);
                        setSelectedItemType('appointment');
                        setTimeout(() => {
                          setShowDetailModal(true);
                        }, 50);
                      }
                    }}>
                    <p className="text-sm font-medium text-gray-900">{apt.patient?.full_name || apt.patient?.user?.full_name}</p>
                    <p className="text-xs text-gray-600">BS. {apt.doctor?.user?.full_name}</p>
                    <div className="mt-2 flex items-center justify-between">
                              <p className="text-xs text-gray-500">
                              {(() => {
                                const dt = parseAppointmentDateTime(apt);
                                return dt ? dt.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }) : (apt.appointment_time || '');
                              })()}
                            </p>
                      <span className={`text-xs px-2 py-1 rounded ${getStatusColor(apt.status)}`}>
                        {getStatusLabel(apt.status)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <DollarSign size={20} /> Hóa đơn chờ thanh toán
                </h3>
                <div className="space-y-3">
                  {invoices.filter(inv => inv.status === 'UNPAID').slice(0, 5).map(inv => (
                    <div key={inv.id} className="p-3 bg-orange-50 rounded-lg border-l-4 border-orange-500 hover:bg-orange-100 cursor-pointer transition"
                      onClick={() => { setSelectedItem(inv); setSelectedItemType('invoice'); setShowPaymentModal(true); }}>
                      <p className="text-sm font-medium text-orange-900">#{inv.invoice_number ?? inv.id}</p>
                      <p className="text-xs text-orange-700">{inv.patient?.user?.full_name}</p>
                      <div className="mt-2 flex items-center justify-between">
                        <p className="text-sm font-semibold text-orange-900">
                          {((inv.total ?? inv.total_amount) ? Number(inv.total ?? inv.total_amount).toLocaleString('vi-VN') : '0')} VND
                        </p>
                        <button className="text-xs bg-orange-600 text-white px-2 py-1 rounded hover:bg-orange-700">
                          Thanh toán
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-lg shadow p-4">
                <h4 className="text-md font-semibold text-gray-900 mb-2">Đơn thuốc chờ lập hoá đơn</h4>
                <div className="space-y-2">
                  {pendingPrescriptions.length === 0 && <div className="text-sm text-gray-500">Không có đơn chờ.</div>}
                  {pendingPrescriptions.slice(0,3).map(p => (
                    <div key={p.id} className="flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-gray-100">
                      <div className="text-sm">Đơn #{p.id} — {p.patient?.user?.full_name || 'N/A'}</div>
                      <div className="flex items-center gap-2">
                        <button onClick={() => { setShowCreateFromPrescriptionModal(true); /* modal will handle selection flow */ }} className="text-sm px-2 py-1 bg-indigo-600 text-white rounded">Mở</button>
                        <button onClick={(e) => { e.stopPropagation(); setModalPrescriptionId(p.id); setShowCreateFromPrescriptionModal(true); setShowPrescriptionDropdown(false); }} className="text-sm px-2 py-1 bg-green-600 text-white rounded">Tạo</button>
                      </div>
                    </div>
                  ))}
                  {pendingPrescriptions.length > 3 && (
                    <div className="text-xs text-gray-500">... và {pendingPrescriptions.length - 3} đơn khác</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'appointments',
      label: 'Lịch hẹn',
      icon: Calendar,
      content: (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Quản lý lịch hẹn</h2>
            <button onClick={() => { setActiveTab('appointments'); setAppointmentInitialValues(null); setShowAddAppointmentModal(true); }} className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
              <Plus size={20} /> Thêm lịch hẹn
            </button>
          </div>

          <div className="mb-4">
            <input
              type="text"
              placeholder="Tìm kiếm bệnh nhân, bác sĩ..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">ID</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Bệnh nhân</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Bác sĩ</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Thời gian</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Trạng thái</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Hành động</th>
                </tr>
              </thead>
              <tbody>
                {filteredAppointments.map(apt => (
                  <tr key={apt.id} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm text-gray-600">#{apt.id}</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{apt.patient?.user?.full_name || apt.patient?.full_name || apt.patient?.user?.email || 'N/A'}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">BS. {apt.doctor?.user?.full_name}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {(() => {
                        const dt = parseAppointmentDateTime(apt);
                        return dt ? dt.toLocaleString('vi-VN') : (apt.appointment_time || apt.appointment_date || '');
                      })()}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`text-xs px-2 py-1 rounded ${getStatusColor(apt.status)}`}>
                        {getStatusLabel(apt.status)}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm">
                      <button className="text-blue-600 hover:text-blue-800 mr-3">
                        <Edit2 size={16} />
                      </button>
                      <button className="text-red-600 hover:text-red-800">
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )
    },
    {
      id: 'patients',
      label: 'Bệnh nhân',
      icon: Users,
      content: (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Danh sách bệnh nhân</h2>
            <button onClick={() => setShowAddPatientModal(true)} className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
              <Plus size={20} /> Thêm bệnh nhân
            </button>
          </div>

          <div className="mb-4">
            <input
              type="text"
              placeholder="Tìm kiếm bệnh nhân..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredPatients.slice(0, 12).map(pt => (
              <div key={pt.id} className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 hover:shadow-lg transition cursor-pointer"
                onClick={() => { setSelectedItem(pt); setSelectedItemType('patient'); setShowDetailModal(true); }}>
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center">
                    <User size={20} className="text-white" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-gray-900">{pt.user?.full_name}</p>
                    <p className="text-xs text-gray-600">ID: {pt.id}</p>
                  </div>
                </div>
                <div className="space-y-2 text-xs text-gray-700">
                  <p className="flex items-center gap-2"><Phone size={14} /> {pt.user?.phone || 'N/A'}</p>
                  <p className="flex items-center gap-2"><Mail size={14} /> {pt.user?.email}</p>
                  <p className="flex items-center gap-2"><MapPin size={14} /> {pt.address || 'N/A'}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )
    },
    {
      id: 'invoices',
      label: 'Hóa đơn',
      icon: FileText,
      content: (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Quản lý hóa đơn</h2>
          </div>

          <div className="mb-4">
            <input
              type="text"
              placeholder="Tìm kiếm hóa đơn..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b bg-gray-50">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Số HĐ</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Bệnh nhân</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Thành tiền</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Trạng thái</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Hành động</th>
                </tr>
              </thead>
              <tbody>
                {filteredInvoices.map(inv => (
                  <tr key={inv.id} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm text-gray-900 font-medium">#{inv.invoice_number ?? inv.id}</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{inv.patient?.user?.full_name}</td>
                    <td className="py-3 px-4 text-sm text-gray-900 font-semibold">
                      {((inv.total ?? inv.total_amount) ? Number(inv.total ?? inv.total_amount).toLocaleString('vi-VN') : '0')} VND
                    </td>
                    <td className="py-3 px-4">
                      <span className={`text-xs px-2 py-1 rounded ${getStatusColor(inv.status)}`}>
                        {getStatusLabel(inv.status)}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm space-x-2">
                      <button className="text-blue-600 hover:text-blue-800" onClick={() => { setSelectedItem(inv); setSelectedItemType('invoice'); setShowDetailModal(true); }}>
                        <Eye size={16} className="inline" />
                      </button>
                      <button className="text-green-600 hover:text-green-800" onClick={() => { setSelectedItem(inv); setSelectedItemType('invoice'); setShowPaymentModal(true); }}>
                        <DollarSign size={16} className="inline" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )
    },
    {
      id: 'doctors',
      label: 'Bác sĩ',
      icon: Stethoscope,
      content: (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Danh sách bác sĩ</h2>
          </div>

          <div className="mb-4">
            <input
              type="text"
              placeholder="Tìm kiếm bác sĩ, chuyên khoa..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredDoctors.slice(0, 12).map(doc => (
              <div key={doc.id} className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4 hover:shadow-lg transition"
                onClick={() => { setSelectedItem(doc); setSelectedItemType('doctor'); setShowDetailModal(true); }}>
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 rounded-full bg-green-500 flex items-center justify-center">
                    <Stethoscope size={20} className="text-white" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-gray-900">BS. {doc.user?.full_name}</p>
                    <p className="text-xs text-gray-600">{doc.specialization}</p>
                  </div>
                </div>
                <div className="space-y-2 text-xs text-gray-700">
                  <p className="flex items-center gap-2"><Phone size={14} /> {doc.user?.phone || 'N/A'}</p>
                  <p className="flex items-center gap-2"><Mail size={14} /> {doc.user?.email}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )
    }
  ];

  if (loading) {
    return (
      <ReceptionistLayout>
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="text-gray-600 mt-4">Đang tải dữ liệu...</p>
          </div>
        </div>
      </ReceptionistLayout>
    );
  }

  return (
    <ReceptionistLayout>
      <div className="flex gap-6">
        {/* Left sidebar */}
        <ReceptionistSidebar
          stats={stats}
          counts={{ patients: patients.length, doctors: doctors.length, pendingInvoices: invoices.filter(i=>i.status==='UNPAID').length, todayAppointments: stats.todayAppointments }}
          pending={pendingPrescriptions.length}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />

        {/* Main content area */}
        <div className="flex-1 space-y-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Chào mừng, {user?.full_name}
          </h1>
          <p className="text-gray-600 mt-2">Quản lý toàn bộ hoạt động lễ tân</p>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-lg shadow">
          <div className="flex items-center justify-between border-b">
            <div className="flex flex-wrap">
              {tabs.map(tab => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => { setActiveTab(tab.id); setSearchTerm(''); setShowPrescriptionDropdown(false); }}
                    className={`flex items-center gap-2 px-6 py-4 border-b-2 font-medium transition ${
                      activeTab === tab.id
                        ? 'border-blue-600 text-blue-600'
                        : 'border-transparent text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    <Icon size={18} />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {/* pending prescriptions quick dropdown button on the tab bar */}
            <div className="relative mr-4">
              <button
                onClick={() => setShowPrescriptionDropdown(prev => !prev)}
                className="flex items-center gap-2 px-3 py-2 rounded-md border bg-white hover:bg-gray-50 text-sm"
              >
                <FileText size={16} />
                <span>Đơn chờ</span>
                <span className="ml-1 inline-flex items-center justify-center px-2 py-0.5 text-xs font-semibold leading-none text-white bg-indigo-600 rounded-full">{pendingPrescriptions.length}</span>
              </button>

              {showPrescriptionDropdown && (
                <div className="absolute right-0 mt-2 w-80 bg-white border rounded shadow-lg z-50">
                  <div className="p-3 border-b font-medium">Đơn thuốc chờ lập hoá đơn</div>
                  <div className="max-h-64 overflow-auto">
                    {pendingPrescriptions.length === 0 && (
                      <div className="p-3 text-sm text-gray-600">Không có đơn chờ.</div>
                    )}
                    {pendingPrescriptions.slice(0, 6).map(p => (
                      <div key={p.id} className="p-3 hover:bg-gray-50 flex justify-between items-start cursor-pointer">
                        <div>
                          <div className="font-medium">Đơn #{p.id}</div>
                          <div className="text-xs text-gray-500">{p.patient?.user?.full_name || 'N/A'}</div>
                        </div>
                        <div className="flex flex-col gap-2">
                          <button onClick={(e) => { e.stopPropagation(); setModalPrescriptionId(p.id); setShowCreateFromPrescriptionModal(true); setShowPrescriptionDropdown(false); }} className="px-2 py-1 bg-green-600 text-white rounded text-xs">Tạo</button>
                          <button onClick={(e) => { e.stopPropagation(); setActiveTab('invoices'); setShowPrescriptionDropdown(false); setShowCreateFromPrescriptionModal(true); /* modal handles selection */ }} className="px-2 py-1 bg-indigo-600 text-white rounded text-xs">Mở</button>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="p-2 text-center border-t">
                    <a href="/receptionist/prescriptions-inbox" className="text-sm text-blue-600">Xem tất cả</a>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="p-6">
            {tabs.find(t => t.id === activeTab)?.content}
          </div>
        </div>
        </div>
      </div>

      {/* Detail Modal */}
      {showDetailModal && selectedItemType === 'appointment' && selectedItem && (
        <AppointmentDetailModal
          appointment={selectedItem}
          onClose={() => { setShowDetailModal(false); setSelectedItem(null); setSelectedItemType(null); }}
          onUpdate={() => {
            fetchAllData();
            setShowDetailModal(false);
          }}
        />
      )}

      {showDetailModal && selectedItemType !== 'appointment' && (
        <DetailModal
          item={selectedItem}
          itemType={selectedItemType}
          onClose={() => { setShowDetailModal(false); setSelectedItem(null); setSelectedItemType(null); }}
          onAction={(action, item) => {
            if (action === 'checkin') {
              setShowDetailModal(false);
              setShowCheckInModal(true);
            } else if (action === 'payment') {
              setShowDetailModal(false);
              setShowPaymentModal(true);
            } else if (action === 'view-schedule') {
              setShowDetailModal(false);
              setScheduleDoctor(item);
              setShowScheduleModal(true);
            }
          }}
        />
      )}

      {showScheduleModal && scheduleDoctor && (
        <ScheduleModal
          doctor={scheduleDoctor}
          onClose={() => { setShowScheduleModal(false); setScheduleDoctor(null); }}
          onSelectTimeslot={({ timeslot, date, doctor }) => {
            // prepare initial values for appointment form
            // timeslot may have start_time or time; normalize
            let timeStr = '';
            if (timeslot.start_time) {
              const d = new Date(timeslot.start_time);
              timeStr = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
            } else if (timeslot.time) {
              // assume 'HH:MM' or full time
              const t = new Date(timeslot.time);
              if (!isNaN(t)) timeStr = t.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
              else timeStr = timeslot.time;
            }

            setAppointmentInitialValues({
              doctor_id: doctor.id || doctor.doctor_id || doctor.user?.id,
              appointment_date: date,
              appointment_time: timeStr,
              // ensure timeslot_id is a number (backend validation requires number)
              timeslot_id: timeslot?.id ? Number(timeslot.id) : (timeslot?.timeslot_id ? Number(timeslot.timeslot_id) : undefined),
              patient_id: (selectedItemType === 'patient' && selectedItem) ? selectedItem.id : undefined,
            });
            setShowScheduleModal(false);
            setScheduleDoctor(null);
            setActiveTab('appointments');
            // open the add appointment modal
            setShowAddAppointmentModal(true);
          }}
        />
      )}

      {/* Add Appointment Modal */}
      {showAddAppointmentModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] overflow-auto">
            <div className="p-4 border-b flex items-center justify-between">
              <h3 className="text-lg font-semibold">Tạo lịch hẹn</h3>
              <button onClick={() => { setShowAddAppointmentModal(false); setAppointmentInitialValues(null); }} className="text-gray-500 hover:text-gray-800">Đóng</button>
            </div>
            <div className="p-4">
              <BookAppointmentForm
                doctors={doctors}
                patients={patients}
                initialValues={appointmentInitialValues}
                allowNewPatient={true}
                onSubmit={async (formData) => {
                  try {
                    await appointmentService.createAppointment(formData);
                    alert('✅ Tạo lịch hẹn thành công');
                    setShowAddAppointmentModal(false);
                    setAppointmentInitialValues(null);
                    // refresh
                    fetchAllData();
                  } catch (err) {
                    alert('❌ Tạo lịch hẹn thất bại: ' + (err?.message || 'Lỗi server'));
                  }
                }}
              />
            </div>
          </div>
        </div>
      )}

        {/* Add Patient Modal */}
        {showAddPatientModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-auto">
              <div className="p-4 border-b flex items-center justify-between">
                <h3 className="text-lg font-semibold">Thêm bệnh nhân</h3>
                <button onClick={() => setShowAddPatientModal(false)} className="text-gray-500 hover:text-gray-800">Đóng</button>
              </div>
              <div className="p-4">
                {patientError && (
                  <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-800 rounded">{patientError}</div>
                )}
                <PatientRegistrationForm
                  onSave={async (formData) => {
                    try {
                      setCreatingPatient(true);
                      setPatientError('');

                      const mapGender = (g) => {
                        if (!g) return undefined;
                        const v = String(g).toLowerCase();
                        if (v === 'nam' || v === 'male') return 'Male';
                        if (v === 'nữ' || v === 'nu' || v === 'female') return 'Female';
                        return 'Other';
                      };

                      // Only include email if receptionist explicitly entered one.
                      const email = formData.email && formData.email.trim().length > 0
                        ? formData.email.trim()
                        : undefined;

                      const payload = {
                        full_name: formData.name,
                        phone: formData.phone,
                        dob: formData.dob || undefined,
                        gender: mapGender(formData.gender),
                        address: formData.address || undefined,
                      };
                      if (email) payload.email = email;

                      // Use quick create endpoint so we do NOT reserve an account/email/password
                      await patientService.createPatientQuick(payload);
                      // refresh lists
                      await fetchAllData();
                      setShowAddPatientModal(false);
                    } catch (err) {
                      console.error('Create patient error', err);
                      const msg = err?.response?.data?.message || err.message || 'Lỗi khi tạo bệnh nhân';
                      setPatientError(msg);
                    } finally {
                      setCreatingPatient(false);
                    }
                  }}
                  onCancel={() => setShowAddPatientModal(false)}
                />
              </div>
            </div>
          </div>
        )}

      {/* Payment Modal */}
      {showPaymentModal && selectedItem && (
        <PaymentModal
          invoice={selectedItem}
          onClose={() => { setShowPaymentModal(false); setSelectedItem(null); }}
          onPaymentSuccess={(paymentData) => {
            alert('✅ Thanh toán thành công: ' + paymentData.paidAmount.toLocaleString('vi-VN') + ' VND');
            // Refresh data
            fetchAllData();
          }}
        />
      )}

      {/* Create invoice from prescription Modal */}
      {showCreateFromPrescriptionModal && (
        <CreateInvoiceFromPrescriptionModal
          initialPrescriptionId={modalPrescriptionId}
          onClose={() => { setShowCreateFromPrescriptionModal(false); setModalPrescriptionId(null); }}
          onInvoiceCreated={(inv) => {
            // open payment modal with created invoice
            setSelectedItem(inv);
            setSelectedItemType('invoice');
            setShowPaymentModal(true);
          }}
        />
      )}

      {/* CheckIn Modal */}
      {showCheckInModal && selectedItem && (
        <CheckInModal
          appointment={selectedItem}
          onClose={() => { setShowCheckInModal(false); setSelectedItem(null); }}
          onCheckInSuccess={(checkInData) => {
            alert('✅ Check-in thành công cho: ' + selectedItem.patient?.user?.full_name);
            // Refresh data
            fetchAllData();
          }}
        />
      )}
    </ReceptionistLayout>
  );
}
