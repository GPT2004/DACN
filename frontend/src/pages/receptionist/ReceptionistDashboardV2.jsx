
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
  User, FileText, Stethoscope, Search, Filter
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

export default function ReceptionistDashboardV2() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [appointments, setAppointments] = useState([]);
  const [patients, setPatients] = useState([]);
  const [invoices, setInvoices] = useState([]);
  const [pendingPrescriptions, setPendingPrescriptions] = useState([]);
  const [doctors, setDoctors] = useState([]);
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
  const [appointmentInitialValues, setAppointmentInitialValues] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [stats, setStats] = useState({
    totalPatients: 0,
    todayAppointments: 0,
    pendingInvoices: 0,
    checkedInToday: 0
  });
  const [appointmentFilter, setAppointmentFilter] = useState('ALL');
  const [patientFilter, setPatientFilter] = useState('ALL');
  const [invoiceFilter, setInvoiceFilter] = useState('ALL');
  const [doctorFilter, setDoctorFilter] = useState('ALL');

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
      setAppointments(aptList);

      // Fetch patients
      const ptsRes = await patientService.getPatients({ limit: 100, page: 1 })
        .catch(() => ({ data: { patients: [] } }));
      let ptList = ptsRes?.data?.patients || [];
      setPatients(ptList);

      // Fetch invoices
      const invRes = await invoiceService.getInvoices({ limit: 100, page: 1 })
        .catch(() => ({ data: { invoices: [] } }));
      let invList = invRes?.data?.invoices || [];
      const mappedInvs = invList.map(inv => {
        const patient = ptList.find(pt => pt.id === Number(inv.patient_id));
        return { ...inv, patient };
      });
      const pendingInvs = mappedInvs.filter(inv => inv.status === 'UNPAID');
      setInvoices(mappedInvs);

      // Fetch prescriptions waiting for invoicing
      const presRes = await prescriptionService.getForInvoicing({ page: 1, limit: 10 }).catch(() => ({ data: { prescriptions: [] } }));
      const presList = presRes?.data?.prescriptions || presRes?.data || [];
      const mappedPres = presList.map(p => {
        const patient = ptList.find(pt => pt.id === Number(p.patient_id));
        return { ...p, patient };
      });
      setPendingPrescriptions(mappedPres);

      // Fetch doctors
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
      // eslint-disable-next-line no-console
      console.error('Error fetching data:', error);
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

  const parseAppointmentDateTime = (apt) => {
    try {
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

  const filteredAppointments = appointments.filter(apt =>
    (apt.patient?.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    apt.doctor?.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase())) &&
    (appointmentFilter === 'ALL' || apt.status === appointmentFilter)
  );

  const filteredPatients = patients.filter(pt =>
    pt.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    pt.user?.email?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const filteredInvoices = invoices.filter(inv => {
    const searchLower = searchTerm.toLowerCase();
    const matchesSearch = 
      String(inv.id).includes(searchLower) ||
      inv.patient?.full_name?.toLowerCase().includes(searchLower) ||
      inv.patient?.user?.full_name?.toLowerCase().includes(searchLower);
    return matchesSearch && (invoiceFilter === 'ALL' || inv.status === invoiceFilter);
  });

  const filteredDoctors = doctors.filter(doc =>
    doc.user?.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.specialization?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const tabs = [
    {
      id: 'overview',
      label: 'Tổng quan',
      icon: TrendingUp,
      content: (
        <div className="space-y-8">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-100 text-sm font-medium">Tổng bệnh nhân</p>
                  <p className="text-3xl font-bold mt-2">{stats.totalPatients}</p>
                </div>
                <Users className="text-blue-200" size={48} />
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-purple-100 text-sm font-medium">Lịch hẹn hôm nay</p>
                  <p className="text-3xl font-bold mt-2">{stats.todayAppointments}</p>
                </div>
                <Calendar className="text-purple-200" size={48} />
              </div>
            </div>

            <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-orange-100 text-sm font-medium">Chờ thanh toán</p>
                  <p className="text-3xl font-bold mt-2">{stats.pendingInvoices}</p>
                </div>
                <DollarSign className="text-orange-200" size={48} />
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-green-100 text-sm font-medium">Đã check-in</p>
                  <p className="text-3xl font-bold mt-2">{stats.checkedInToday}</p>
                </div>
                <CheckCircle2 className="text-green-200" size={48} />
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-white rounded-xl shadow-sm border p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <button
                onClick={() => { setActiveTab('appointments'); setAppointmentInitialValues(null); setShowAddAppointmentModal(true); }}
                className="flex flex-col items-center p-4 bg-blue-50 rounded-lg hover:bg-blue-100 transition"
              >
                <Plus className="text-blue-600 mb-2" size={24} />
                <span className="text-sm font-medium text-blue-900">Tạo lịch hẹn</span>
              </button>
              <button
                onClick={() => setShowAddPatientModal(true)}
                className="flex flex-col items-center p-4 bg-green-50 rounded-lg hover:bg-green-100 transition"
              >
                <User className="text-green-600 mb-2" size={24} />
                <span className="text-sm font-medium text-green-900">Thêm bệnh nhân</span>
              </button>
              <button
                onClick={() => setActiveTab('prescriptions')}
                className="flex flex-col items-center p-4 bg-purple-50 rounded-lg hover:bg-purple-100 transition"
              >
                <FileText className="text-purple-600 mb-2" size={24} />
                <span className="text-sm font-medium text-purple-900">Tạo hóa đơn</span>
              </button>
              <button
                onClick={() => setActiveTab('appointments')}
                className="flex flex-col items-center p-4 bg-orange-50 rounded-lg hover:bg-orange-100 transition"
              >
                <Calendar className="text-orange-600 mb-2" size={24} />
                <span className="text-sm font-medium text-orange-900">Xem lịch hẹn</span>
              </button>
            </div>
          </div>

          {/* Recent Appointments & Pending Invoices */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Calendar size={20} className="text-blue-600" /> Lịch hẹn sắp tới
              </h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {appointments.slice(0, 10).map(apt => (
                  <div key={apt.id} className="p-4 bg-gray-50 rounded-lg border-l-4 border-blue-500 hover:bg-gray-100 cursor-pointer transition"
                    onClick={async () => {
                      try {
                        const fullApt = await appointmentService.getAppointmentById(apt.id);
                        const serverPayload = fullApt?.data ?? fullApt;
                        const appointmentObj = serverPayload?.data ?? serverPayload;
                        setSelectedItem(appointmentObj || apt);
                        setSelectedItemType('appointment');
                        setTimeout(() => setShowDetailModal(true), 50);
                      } catch (err) {
                        // eslint-disable-next-line no-console
                        console.error('Error fetching appointment:', err);
                        setSelectedItem(apt);
                        setSelectedItemType('appointment');
                        setTimeout(() => setShowDetailModal(true), 50);
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

            <div className="space-y-6">
              <div className="bg-white rounded-xl shadow-sm border p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <DollarSign size={20} className="text-orange-600" /> Hóa đơn chờ thanh toán
                </h3>
                <div className="space-y-3">
                  {invoices.filter(inv => inv.status === 'UNPAID').slice(0, 5).map(inv => (
                    <div key={inv.id} className="p-3 bg-orange-50 rounded-lg border-l-4 border-orange-500 hover:bg-orange-100 cursor-pointer transition"
                      onClick={() => { setSelectedItem(inv); setSelectedItemType('invoice'); setShowPaymentModal(true); }}>
                      <p className="text-sm font-medium text-orange-900">#{inv.invoice_number ?? inv.id}</p>
                      <p className="text-xs text-orange-700">{inv.patient?.full_name || inv.patient?.user?.full_name}</p>
                      <div className="mt-2 flex items-center justify-between">
                        <p className="text-sm font-semibold text-orange-900">
                          {((inv.total ?? inv.total_amount) ? Number(inv.total ?? inv.total_amount).toLocaleString('vi-VN') : '0')} VND
                        </p>
                        <button className="text-xs bg-orange-600 text-white px-3 py-1 rounded hover:bg-orange-700">
                          Thanh toán
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-sm border p-4">
                <h4 className="text-md font-semibold text-gray-900 mb-2 flex items-center gap-2">
                  <FileText size={16} className="text-purple-600" /> Đơn thuốc chờ lập hóa đơn
                </h4>
                <div className="space-y-2">
                  {pendingPrescriptions.length === 0 && <div className="text-sm text-gray-500">Không có đơn chờ.</div>}
                  {pendingPrescriptions.slice(0,3).map(p => (
                    <div key={p.id} className="flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-gray-100">
                      <div className="text-sm">Đơn #{p.id} — {p.patient?.full_name || p.patient?.user?.full_name || p.patient_name || 'Bệnh nhân chưa xác định'}</div>
                      <div className="flex items-center gap-2">
                        <button onClick={() => { setShowCreateFromPrescriptionModal(true); }} className="text-sm px-2 py-1 bg-indigo-600 text-white rounded">Mở</button>
                        <button onClick={(e) => { e.stopPropagation(); setModalPrescriptionId(p.id); setShowCreateFromPrescriptionModal(true); }} className="text-sm px-2 py-1 bg-green-600 text-white rounded">Tạo</button>
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
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Quản lý lịch hẹn</h2>
            <button onClick={() => { setActiveTab('appointments'); setAppointmentInitialValues(null); setShowAddAppointmentModal(true); }} className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
              <Plus size={20} /> Thêm lịch hẹn
            </button>
          </div>

          <div className="mb-4 flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-3 text-gray-400" size={16} />
              <input
                type="text"
                placeholder="Tìm kiếm bệnh nhân, bác sĩ..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <select
              value={appointmentFilter}
              onChange={(e) => setAppointmentFilter(e.target.value)}
              className="px-4 py-2 border rounded-lg"
            >
              <option value="ALL">Tất cả trạng thái</option>
              <option value="PENDING">Chờ xác nhận</option>
              <option value="CONFIRMED">Đã xác nhận</option>
              <option value="CHECKED_IN">Đã check-in</option>
              <option value="IN_PROGRESS">Đang khám</option>
              <option value="COMPLETED">Hoàn thành</option>
              <option value="CANCELLED">Đã hủy</option>
            </select>
          </div>

          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full">
              <thead className="sticky top-0 bg-gray-50">
                <tr className="border-b bg-gray-50">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">ID</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Bệnh nhân</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Bác sĩ</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Email</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">SĐT</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Trạng thái</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Hành động</th>
                </tr>
              </thead>
              <tbody>
                {filteredAppointments.map(apt => (
                  <tr key={apt.id} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm text-gray-600">#{apt.id}</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{apt.patient?.user?.full_name || apt.patient?.full_name || apt.patient?.user?.email || 'Bệnh nhân chưa xác định'}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">BS. {apt.doctor?.user?.full_name}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">{apt.patient?.user?.email || apt.patient?.email || '-'}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">{apt.patient?.user?.phone || apt.patient?.phone || '-'}</td>
                    <td className="py-3 px-4">
                      <span className={`text-xs px-2 py-1 rounded ${getStatusColor(apt.status)}`}>
                        {getStatusLabel(apt.status)}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm">
                      <button 
                        className="text-blue-600 hover:text-blue-800 mr-2" 
                        onClick={() => { setSelectedItem(apt); setSelectedItemType('appointment'); setShowDetailModal(true); }}
                        title="Xem chi tiết"
                      >
                        <Edit2 size={16} />
                      </button>
                      {(apt.status === 'CONFIRMED' || apt.status === 'PENDING') && (
                        <button 
                          className="text-green-600 hover:text-green-800 mr-2" 
                          onClick={() => { setSelectedItem(apt); setShowCheckInModal(true); }}
                          title="Check-in"
                        >
                          <CheckCircle2 size={16} />
                        </button>
                      )}
                      <button 
                        className="text-red-600 hover:text-red-800" 
                        onClick={async () => {
                          // eslint-disable-next-line no-restricted-globals
                          if (confirm('Bạn có chắc muốn xóa lịch hẹn này?')) {
                            try {
                              await appointmentService.deleteAppointment(apt.id);
                              fetchAllData();
                            } catch (err) {
                              alert('Lỗi xóa lịch hẹn: ' + (err?.message || 'Lỗi server'));
                            }
                          }
                        }}
                        title="Xóa"
                      >
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
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Danh sách bệnh nhân</h2>
            <button onClick={() => setShowAddPatientModal(true)} className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
              <Plus size={20} /> Thêm bệnh nhân
            </button>
          </div>

          <div className="mb-4 flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-3 text-gray-400" size={16} />
              <input
                type="text"
                placeholder="Tìm kiếm bệnh nhân..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <button className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50">
              <Filter size={16} />
              Lọc
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-h-96 overflow-y-auto">
            {filteredPatients.slice(0, 12).map(pt => (
              <div key={pt.id} className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 hover:shadow-lg transition cursor-pointer border"
                onClick={() => { setSelectedItem(pt); setSelectedItemType('patient'); setShowDetailModal(true); }}>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 rounded-full bg-blue-500 flex items-center justify-center">
                    <User size={24} className="text-white" />
                  </div>
                  <div>
                    <p className="text-lg font-semibold text-gray-900">{pt.user?.full_name}</p>
                    <p className="text-sm text-gray-600">ID: {pt.id}</p>
                  </div>
                </div>
                <div className="space-y-3 text-sm text-gray-700">
                  <p className="flex items-center gap-2"><Phone size={16} /> {pt.user?.phone || 'Chưa cập nhật'}</p>
                  <p className="flex items-center gap-2"><Mail size={16} /> {pt.user?.email}</p>
                  <p className="flex items-center gap-2"><MapPin size={16} /> {pt.address || 'Chưa cập nhật'}</p>
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
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Quản lý hóa đơn</h2>
          </div>

          <div className="mb-4 flex gap-4 items-center">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-3 text-gray-400" size={16} />
              <input
                type="text"
                placeholder="Tìm kiếm theo số HĐ hoặc tên bệnh nhân..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <select
              value={invoiceFilter}
              onChange={(e) => setInvoiceFilter(e.target.value)}
              className="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="ALL">Tất cả trạng thái</option>
              <option value="PAID">Đã thanh toán</option>
              <option value="UNPAID">Chưa thanh toán</option>
            </select>
            <span className="text-sm text-gray-600">({filteredInvoices.length} hóa đơn)</span>
          </div>

          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            {filteredInvoices.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                Không có hóa đơn nào{searchTerm ? ` phù hợp với "${searchTerm}"` : ''}
              </div>
            ) : (
              <table className="w-full">
                <thead className="sticky top-0 bg-gray-50">
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
                      <td className="py-3 px-4 text-sm text-gray-900 font-medium">#{inv.id}</td>
                      <td className="py-3 px-4 text-sm text-gray-900">{inv.patient?.full_name || inv.patient?.user?.full_name}</td>
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
            )}
          </div>
        </div>
      )
    },
    {
      id: 'doctors',
      label: 'Bác sĩ',
      icon: Stethoscope,
      content: (
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Danh sách bác sĩ</h2>
          </div>

          <div className="mb-4 flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-3 text-gray-400" size={16} />
              <input
                type="text"
                placeholder="Tìm kiếm bác sĩ, chuyên khoa..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <button className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50">
              <Filter size={16} />
              Lọc
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-h-96 overflow-y-auto">
            {filteredDoctors.slice(0, 12).map(doc => (
              <div key={doc.id} className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 hover:shadow-lg transition border"
                onClick={() => { setSelectedItem(doc); setSelectedItemType('doctor'); setShowDetailModal(true); }}>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 rounded-full bg-green-500 flex items-center justify-center">
                    <Stethoscope size={24} className="text-white" />
                  </div>
                  <div>
                    <p className="text-lg font-semibold text-gray-900">BS. {doc.user?.full_name}</p>
                    <p className="text-sm text-gray-600">{doc.specialization}</p>
                  </div>
                </div>
                <div className="space-y-3 text-sm text-gray-700">
                  <p className="flex items-center gap-2"><Phone size={16} /> {doc.user?.phone || 'Chưa cập nhật'}</p>
                  <p className="flex items-center gap-2"><Mail size={16} /> {doc.user?.email}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )
    },
    {
      id: 'prescriptions',
      label: 'Đơn thuốc',
      icon: FileText,
      content: (
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Đơn thuốc chờ lập hóa đơn</h2>
          </div>

          <div className="mb-4 flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-3 text-gray-400" size={16} />
              <input
                type="text"
                placeholder="Tìm kiếm đơn thuốc..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-h-96 overflow-y-auto">
            {pendingPrescriptions.map(p => (
              <div key={p.id} className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-6 hover:shadow-lg transition border">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 rounded-full bg-purple-500 flex items-center justify-center">
                    <FileText size={24} className="text-white" />
                  </div>
                  <div>
                    <p className="text-lg font-semibold text-gray-900">Đơn #{p.id}</p>
                    <p className="text-sm text-gray-600">Bệnh nhân: {p.patient?.full_name || p.patient?.user?.full_name || p.patient_name || 'Chưa xác định'}</p>
                  </div>
                </div>
                <div className="space-y-3 text-sm text-gray-700">
                  <p className="flex items-center gap-2"><User size={16} /> BS. {p.doctor?.user?.full_name}</p>
                  <p className="flex items-center gap-2"><DollarSign size={16} /> {p.total_amount ? Number(p.total_amount).toLocaleString('vi-VN') : '0'} VND</p>
                  <p className="text-xs text-gray-500">Tạo: {new Date(p.created_at).toLocaleDateString('vi-VN')}</p>
                </div>
                <div className="mt-4 flex gap-2">
                  <button onClick={() => { setModalPrescriptionId(p.id); setShowCreateFromPrescriptionModal(true); }} className="flex-1 px-3 py-2 bg-green-600 text-white rounded text-sm hover:bg-green-700">Tạo hóa đơn</button>
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
      <div className="flex gap-8 min-h-screen bg-gray-50">
        {/* Sidebar */}
        <ReceptionistSidebar
          stats={stats}
          counts={{ patients: patients.length, doctors: doctors.length, pendingInvoices: invoices.filter(i=>i.status==='UNPAID').length, todayAppointments: stats.todayAppointments }}
          pending={pendingPrescriptions.length}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />

        {/* Main content */}
        <div className="flex-1 p-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900">
              Chào mừng, {user?.full_name}
            </h1>
            <p className="text-gray-600 mt-2">Quản lý toàn bộ hoạt động lễ tân Phòng khám đa khoa</p>
          </div>

          {/* Tabs */}
          <div className="bg-white rounded-xl shadow-sm border mb-6">
            <div className="flex items-center justify-between border-b px-6 py-4">
              <div className="flex flex-wrap gap-2">
                {tabs.map(tab => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => { setActiveTab(tab.id); setSearchTerm(''); setAppointmentFilter('ALL'); setPatientFilter('ALL'); setInvoiceFilter('ALL'); setDoctorFilter('ALL'); }}
                      className={`flex items-center gap-3 px-6 py-3 rounded-lg font-medium transition ${
                        activeTab === tab.id
                          ? 'bg-blue-600 text-white shadow-md'
                          : 'text-gray-600 hover:bg-gray-100'
                      }`}
                    >
                      <Icon size={20} />
                      {tab.label}
                      {tab.id === 'prescriptions' && pendingPrescriptions.length > 0 && (
                        <span className="ml-2 inline-flex items-center justify-center px-2 py-0.5 text-xs font-semibold leading-none text-white bg-red-600 rounded-full">
                          {pendingPrescriptions.length}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="p-6">
              {tabs.find(t => t.id === activeTab)?.content}
            </div>
          </div>
        </div>
      </div>

      {/* Modals */}
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
            let timeStr = '';
            if (timeslot.start_time) {
              const d = new Date(timeslot.start_time);
              timeStr = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
            } else if (timeslot.time) {
              const t = new Date(timeslot.time);
              if (!isNaN(t)) timeStr = t.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
              else timeStr = timeslot.time;
            }

            setAppointmentInitialValues({
              doctor_id: doctor.id || doctor.doctor_id || doctor.user?.id,
              appointment_date: date,
              appointment_time: timeStr,
              timeslot_id: timeslot?.id ? Number(timeslot.id) : (timeslot?.timeslot_id ? Number(timeslot.timeslot_id) : undefined),
              patient_id: (selectedItemType === 'patient' && selectedItem) ? selectedItem.id : undefined,
            });
            setShowScheduleModal(false);
            setScheduleDoctor(null);
            setActiveTab('appointments');
            setShowAddAppointmentModal(true);
          }}
        />
      )}

      {/* Add Appointment Modal */}
      {showAddAppointmentModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-4xl max-h-[90vh] overflow-auto">
            <div className="p-6 border-b flex items-center justify-between">
              <h3 className="text-xl font-semibold">Tạo lịch hẹn</h3>
              <button onClick={() => { setShowAddAppointmentModal(false); setAppointmentInitialValues(null); }} className="text-gray-500 hover:text-gray-800">
                <span className="text-2xl">&times;</span>
              </button>
            </div>
            <div className="p-6">
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
          <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-auto">
            <div className="p-6 border-b flex items-center justify-between">
              <h3 className="text-xl font-semibold">Thêm bệnh nhân</h3>
              <button onClick={() => setShowAddPatientModal(false)} className="text-gray-500 hover:text-gray-800">
                <span className="text-2xl">&times;</span>
              </button>
            </div>
            <div className="p-6">
              {patientError && (
                <div className="mb-4 p-4 bg-red-50 border border-red-200 text-red-800 rounded-lg">{patientError}</div>
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

                    await patientService.createPatientQuick(payload);
                    await fetchAllData();
                    setShowAddPatientModal(false);
                  } catch (err) {
                    // eslint-disable-next-line no-console
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
            setActiveTab('invoices');
            setInvoiceFilter('ALL');
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
            setSelectedItem(inv);
            setSelectedItemType('invoice');
            setActiveTab('invoices');
            setInvoiceFilter('ALL');
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
            fetchAllData();
          }}
        />
      )}
    </ReceptionistLayout>
  );
}
