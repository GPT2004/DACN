import React from 'react';
import { X, Phone, Mail, MapPin, Calendar, User, Stethoscope, DollarSign, FileText } from 'lucide-react';

/**
 * DetailModal - Hiển thị chi tiết các entity (Appointment, Patient, Invoice, Doctor)
 */
export default function DetailModal({ item, itemType, onClose, onAction }) {
  if (!item) return null;

  const renderAppointmentDetails = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-gray-600">ID Lịch hẹn</label>
          <p className="text-sm font-semibold text-gray-900">#{item.id}</p>
        </div>
        <div>
          <label className="text-xs text-gray-600">Trạng thái</label>
          <p className={`text-sm font-semibold px-2 py-1 rounded w-fit ${
            item.status === 'COMPLETED' ? 'bg-green-100 text-green-800' :
            item.status === 'CHECKED_IN' ? 'bg-purple-100 text-purple-800' :
            item.status === 'CONFIRMED' ? 'bg-blue-100 text-blue-800' :
            'bg-yellow-100 text-yellow-800'
          }`}>
            {item.status}
          </p>
        </div>
      </div>

      <div className="border-t pt-4">
        <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <User size={18} /> Thông tin bệnh nhân
        </h4>
        <div className="space-y-2 text-sm">
          <p><span className="text-gray-600">Tên:</span> <span className="font-medium">{item.patient?.user?.full_name}</span></p>
          <p className="flex items-center gap-2 text-gray-600">
            <Phone size={14} /> {item.patient?.user?.phone || 'N/A'}
          </p>
          <p className="flex items-center gap-2 text-gray-600">
            <Mail size={14} /> {item.patient?.user?.email}
          </p>
          <p className="flex items-center gap-2 text-gray-600">
            <MapPin size={14} /> {item.patient?.address || 'N/A'}
          </p>
        </div>
      </div>

      <div className="border-t pt-4">
        <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Stethoscope size={18} /> Thông tin bác sĩ
        </h4>
        <div className="space-y-2 text-sm">
          <p><span className="text-gray-600">Tên:</span> <span className="font-medium">BS. {item.doctor?.user?.full_name}</span></p>
          <p><span className="text-gray-600">Chuyên khoa:</span> <span className="font-medium">{item.doctor?.specialization}</span></p>
          <p className="flex items-center gap-2 text-gray-600">
            <Phone size={14} /> {item.doctor?.user?.phone || 'N/A'}
          </p>
        </div>
      </div>

      <div className="border-t pt-4">
        <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Calendar size={18} /> Thông tin lịch hẹn
        </h4>
        <div className="space-y-2 text-sm">
              {(() => {
                // Prefer combined datetime from backend; otherwise combine date + time as local time
                try {
                  let dt = null;
                  if (item.appointment_datetime) dt = new Date(item.appointment_datetime);
                  else if (item.appointment_date && item.appointment_time) {
                    // Parse time component
                    const timeRaw = item.appointment_time;
                    let hours = 0, minutes = 0, seconds = 0;
                    
                    if (typeof timeRaw === 'string' && /^\d{2}:\d{2}(:\d{2})?/.test(timeRaw)) {
                      const parts = timeRaw.split(':');
                      hours = parseInt(parts[0], 10);
                      minutes = parseInt(parts[1], 10);
                      seconds = parts[2] ? parseInt(parts[2], 10) : 0;
                    }
                    
                    const dateOnly = new Date(item.appointment_date);
                    const y = dateOnly.getFullYear();
                    const m = dateOnly.getMonth();
                    const d = dateOnly.getDate();
                    dt = new Date(y, m, d, hours, minutes, seconds);
                  }
                  else if (item.appointment_time) dt = new Date(item.appointment_time);
                  else if (item.appointment_date) dt = new Date(item.appointment_date);

                  const dateStr = dt && !isNaN(dt.getTime()) ? dt.toLocaleDateString('vi-VN') : (item.appointment_date ? new Date(item.appointment_date).toLocaleDateString('vi-VN') : 'N/A');
                  const timeStr = dt && !isNaN(dt.getTime()) ? dt.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }) : (item.appointment_time || 'N/A');
                  return (
                    <>
                      <p><span className="text-gray-600">Ngày:</span> <span className="font-medium">{dateStr}</span></p>
                      <p><span className="text-gray-600">Giờ:</span> <span className="font-medium">{timeStr}</span></p>
                    </>
                  );
                } catch (e) {
                  return (
                    <>
                      <p><span className="text-gray-600">Ngày:</span> <span className="font-medium">{item.appointment_date ? new Date(item.appointment_date).toLocaleDateString('vi-VN') : 'N/A'}</span></p>
                      <p><span className="text-gray-600">Giờ:</span> <span className="font-medium">{item.appointment_time || 'N/A'}</span></p>
                    </>
                  );
                }
              })()}
          <p><span className="text-gray-600">Lý do khám:</span> <span className="font-medium">{item.reason || 'N/A'}</span></p>
          <p><span className="text-gray-600">Phòng:</span> <span className="font-medium">{item.room?.name || 'N/A'}</span></p>
        </div>
      </div>

      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
        <p className="text-xs text-blue-800"><strong>Ghi chú:</strong> {item.notes || 'Không có ghi chú'}</p>
      </div>

      <div className="flex gap-3 pt-4 border-t">
        <button onClick={() => onAction('checkin', item)} className="flex-1 bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 text-sm font-medium">
          Check-in
        </button>
        <button onClick={() => onAction('edit', item)} className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 text-sm font-medium">
          Chỉnh sửa
        </button>
        <button onClick={() => onAction('delete', item)} className="flex-1 bg-red-600 text-white py-2 px-4 rounded-lg hover:bg-red-700 text-sm font-medium">
          Hủy
        </button>
      </div>
    </div>
  );

  const renderPatientDetails = () => (
    <div className="space-y-4">
      <div>
        <label className="text-xs text-gray-600">ID Bệnh nhân</label>
        <p className="text-sm font-semibold text-gray-900">#{item.id}</p>
      </div>

      <div className="border-t pt-4">
        <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <User size={18} /> Thông tin cá nhân
        </h4>
        <div className="space-y-2 text-sm">
          <p><span className="text-gray-600">Họ tên:</span> <span className="font-medium">{item.user?.full_name}</span></p>
          <p className="flex items-center gap-2 text-gray-600">
            <Phone size={14} /> {item.user?.phone || 'N/A'}
          </p>
          <p className="flex items-center gap-2 text-gray-600">
            <Mail size={14} /> {item.user?.email}
          </p>
          <p className="flex items-center gap-2 text-gray-600">
            <MapPin size={14} /> {item.address || 'N/A'}
          </p>
          <p><span className="text-gray-600">Giới tính:</span> <span className="font-medium">{item.gender || 'N/A'}</span></p>
          <p><span className="text-gray-600">Ngày sinh:</span> <span className="font-medium">{item.dob ? new Date(item.dob).toLocaleDateString('vi-VN') : 'N/A'}</span></p>
        </div>
      </div>

      <div className="bg-green-50 p-4 rounded-lg border border-green-200">
        <p className="text-xs text-green-800"><strong>Trạng thái:</strong> {item.user?.is_active ? 'Hoạt động' : 'Không hoạt động'}</p>
      </div>

      <div className="flex gap-3 pt-4 border-t">
        <button onClick={() => onAction('view-records', item)} className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 text-sm font-medium">
          Xem hồ sơ
        </button>
        <button onClick={() => onAction('book-appointment', item)} className="flex-1 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 text-sm font-medium">
          Đặt lịch hẹn
        </button>
      </div>
    </div>
  );

  const renderInvoiceDetails = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-gray-600">Số hóa đơn</label>
          <p className="text-sm font-semibold text-gray-900">#{item.invoice_number ?? item.id}</p>
        </div>
        <div>
          <label className="text-xs text-gray-600">Trạng thái</label>
          <p className={`text-sm font-semibold px-2 py-1 rounded w-fit ${
            item.status === 'PAID' ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'
          }`}>
            {item.status === 'PAID' ? 'Đã thanh toán' : 'Chưa thanh toán'}
          </p>
        </div>
      </div>

      <div className="border-t pt-4">
        <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <User size={18} /> Bệnh nhân
        </h4>
        <div className="space-y-2 text-sm">
          <p><span className="text-gray-600">Tên:</span> <span className="font-medium">{item.patient?.user?.full_name}</span></p>
          <p className="flex items-center gap-2 text-gray-600">
            <Phone size={14} /> {item.patient?.user?.phone}
          </p>
        </div>
      </div>

      <div className="border-t pt-4">
        <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <DollarSign size={18} /> Chi tiết thanh toán
        </h4>
        <div className="space-y-2 text-sm">
          {(() => {
            const items = Array.isArray(item.items) ? item.items : [];
            const totalItemsSum = items.reduce((s, it) => s + (Number(it.amount || it.value || 0)), 0);
            const consultItem = items.find(i => /consult|khám|phí khám|phí khám bệnh|exam/i.test(String(i.type || i.description || '')));
            const consultFee = consultItem ? Number(consultItem.amount || consultItem.value || 0) : Number(item?.consultation_fee ?? item?.consultationFee ?? item?.doctor?.consultation_fee ?? item?.appointment?.doctor?.consultation_fee ?? 0);
            let medicinesSubtotal = 0;
            if (items.length > 0) {
              if (consultItem) {
                medicinesSubtotal = Math.max(0, totalItemsSum - (Number(consultItem.amount || 0)));
              } else if (item.subtotal) {
                medicinesSubtotal = Number(item.subtotal || 0);
              } else {
                medicinesSubtotal = totalItemsSum;
              }
            } else {
              medicinesSubtotal = Number(item.subtotal || 0);
            }

            const totalAmount = Number(item?.total_amount ?? item?.total ?? (medicinesSubtotal + consultFee) ?? 0);
            const paid = Number(item?.paid_amount || 0);
            const owed = Math.max(0, totalAmount - paid);

            return (
              <>
                <div className="flex justify-between"><span className="text-gray-600">Phí khám:</span> <span className="font-semibold">{consultFee.toLocaleString('vi-VN')} VND</span></div>
                <div className="flex justify-between"><span className="text-gray-600">Thuốc:</span> <span className="font-semibold">{medicinesSubtotal.toLocaleString('vi-VN')} VND</span></div>
                <div className="flex justify-between"><span className="text-gray-600">Tổng tiền:</span> <span className="font-medium">{totalAmount.toLocaleString('vi-VN')} VND</span></div>
                <div className="flex justify-between"><span className="text-gray-600">Đã thanh toán:</span> <span className="font-medium">{paid.toLocaleString('vi-VN')} VND</span></div>
                <div className="flex justify-between border-t pt-2"><span className="text-gray-600 font-semibold">Còn nợ:</span> <span className="font-bold">{owed.toLocaleString('vi-VN')} VND</span></div>
              </>
            );
          })()}
        </div>
      </div>

      {item.status === 'UNPAID' && (
        <div className="flex gap-3 pt-4 border-t">
          <button onClick={() => onAction('payment', item)} className="flex-1 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 text-sm font-medium">
            Xử lý thanh toán
          </button>
          <button onClick={() => onAction('print', item)} className="flex-1 bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 text-sm font-medium">
            In hóa đơn
          </button>
        </div>
      )}
    </div>
  );

  const renderDoctorDetails = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-gray-600">ID Bác sĩ</label>
          <p className="text-sm font-semibold text-gray-900">#{item.id}</p>
        </div>
        <div>
          <label className="text-xs text-gray-600">Chuyên khoa</label>
          <p className="text-sm font-semibold text-gray-900">{item.specialization}</p>
        </div>
      </div>

      <div className="border-t pt-4">
        <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Stethoscope size={18} /> Thông tin cá nhân
        </h4>
        <div className="space-y-2 text-sm">
          <p><span className="text-gray-600">Họ tên:</span> <span className="font-medium">BS. {item.user?.full_name}</span></p>
          <p className="flex items-center gap-2 text-gray-600">
            <Phone size={14} /> {item.user?.phone || 'N/A'}
          </p>
          <p className="flex items-center gap-2 text-gray-600">
            <Mail size={14} /> {item.user?.email}
          </p>
          <p><span className="text-gray-600">Giấy phép:</span> <span className="font-medium">{item.license_number || 'N/A'}</span></p>
        </div>
      </div>

      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
        <p className="text-xs text-blue-800"><span className="font-semibold">Đánh giá:</span> ⭐ {item.rating || 'N/A'}/5</p>
      </div>

      <div className="flex gap-3 pt-4 border-t">
        <button onClick={() => onAction('view-schedule', item)} className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 text-sm font-medium">
          Xem lịch làm việc
        </button>
        <button onClick={() => onAction('contact', item)} className="flex-1 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 text-sm font-medium">
          Liên hệ
        </button>
      </div>
    </div>
  );

  const detailsMap = {
    appointment: renderAppointmentDetails,
    patient: renderPatientDetails,
    invoice: renderInvoiceDetails,
    doctor: renderDoctorDetails
  };

  const renderDetails = detailsMap[itemType] || (() => <pre>{JSON.stringify(item, null, 2)}</pre>);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-gradient-to-r from-blue-50 to-blue-100 p-6 border-b flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-900">
            {itemType === 'appointment' ? '📅 Chi tiết lịch hẹn' :
             itemType === 'patient' ? '👤 Chi tiết bệnh nhân' :
             itemType === 'invoice' ? '🧾 Chi tiết hóa đơn' :
             itemType === 'doctor' ? '⚕️ Chi tiết bác sĩ' :
             'Chi tiết'}
          </h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 transition">
            <X size={24} />
          </button>
        </div>

        <div className="p-6">
          {renderDetails()}
        </div>
      </div>
    </div>
  );
}
