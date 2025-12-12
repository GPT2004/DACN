import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import 'dayjs/locale/vi';

dayjs.extend(relativeTime);
dayjs.locale('vi');

/**
 * Format relative time (e.g., "2 giờ trước")
 */
export const formatRelativeTime = (date) => {
  if (!date) return '';
  return dayjs(date).fromNow();
};

/**
 * Format patient name with prefix
 */
export const formatPatientName = (patient) => {
  if (!patient) return '';
  const gender = patient.gender || patient.user?.gender;
  const name = patient.full_name || patient.user?.full_name;
  
  if (gender === 'Male') return `Anh ${name}`;
  if (gender === 'Female') return `Chị ${name}`;
  return name;
};

/**
 * Format doctor name with title
 */
export const formatDoctorName = (doctor) => {
  if (!doctor) return '';
  const name = doctor.full_name || doctor.user?.full_name;
  return `BS. ${name}`;
};

/**
 * Format medical record title
 */
export const formatMedicalRecordTitle = (record) => {
  if (!record) return '';
  const date = dayjs(record.created_at).format('DD/MM/YYYY');
  return `Bệnh án ngày ${date}`;
};

/**
 * Format prescription title
 */
export const formatPrescriptionTitle = (prescription) => {
  if (!prescription) return '';
  const date = dayjs(prescription.created_at).format('DD/MM/YYYY');
  return `Đơn thuốc ngày ${date}`;
};

/**
 * Format appointment title
 */
export const formatAppointmentTitle = (appointment) => {
  if (!appointment) return '';
  const date = dayjs(appointment.appointment_date).format('DD/MM/YYYY');
  const time = appointment.appointment_time;
  return `Khám ngày ${date} - ${time}`;
};

/**
 * Format address
 */
export const formatAddress = (address) => {
  if (!address) return '';
  const parts = [];
  
  if (address.street) parts.push(address.street);
  if (address.ward) parts.push(address.ward);
  if (address.district) parts.push(address.district);
  if (address.city) parts.push(address.city);
  
  return parts.join(', ');
};

/**
 * Format emergency contact
 */
export const formatEmergencyContact = (contact) => {
  if (!contact) return '';
  const parts = [];
  
  if (contact.name) parts.push(contact.name);
  if (contact.relationship) parts.push(`(${contact.relationship})`);
  if (contact.phone) parts.push(`- ${contact.phone}`);
  
  return parts.join(' ');
};

/**
 * Format BMI
 */
export const formatBMI = (bmi) => {
  if (!bmi) return '';
  
  let category = '';
  if (bmi < 18.5) category = 'Thiếu cân';
  else if (bmi < 25) category = 'Bình thường';
  else if (bmi < 30) category = 'Thừa cân';
  else category = 'Béo phì';
  
  return `${bmi.toFixed(1)} (${category})`;
};

/**
 * Format blood pressure
 */
export const formatBloodPressure = (systolic, diastolic) => {
  if (!systolic || !diastolic) return '';
  return `${systolic}/${diastolic} mmHg`;
};

/**
 * Format dosage
 */
export const formatDosage = (item) => {
  if (!item) return '';
  const parts = [];
  
  if (item.dosage) parts.push(item.dosage);
  if (item.quantity) parts.push(`x${item.quantity} ${item.unit || 'viên'}`);
  
  return parts.join(' - ');
};

/**
 * Format medicine instructions
 */
export const formatMedicineInstructions = (item) => {
  if (!item || !item.instructions) return 'Không có hướng dẫn';
  return item.instructions;
};