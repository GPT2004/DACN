import React from 'react';

export default function Contact() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <h3 className="text-lg font-semibold mb-2">Về Phòng Khám Tư Nhân</h3>
        <p className="text-sm text-gray-700">Phòng khám tư nhân thành lập với mục đích cung cấp dịch vụ khám chữa bệnh chuyên nghiệp, chất lượng cao. Chúng tôi sở hữu đội ngũ bác sĩ chuyên khoa có kinh nghiệm nhiều năm và trang thiết bị y tế hiện đại. Cam kết mang lại trải nghiệm khám chữa bệnh tốt nhất cho bệnh nhân.</p>
        <ul className="mt-4 text-sm text-gray-700 space-y-1">
          <li>⦿ Địa chỉ: 123 Đường Hồng Bàng, Quận Hải Châu, TP. Đà Nẵng</li>
          <li>⦿ Điện thoại: (0236) 1234 5678</li>
          <li>⦿ Giờ làm việc: 08:00 - 18:00 (Thứ 2 - Thứ 7), 09:00 - 12:00 (Chủ nhật)</li>
        </ul>
      </div>
      <div>
        <h3 className="text-lg font-semibold mb-2">Liên hệ nhanh</h3>
        <form className="space-y-3" onSubmit={(e)=>{e.preventDefault(); alert('Cảm ơn, chúng tôi sẽ liên hệ lại sớm.')}}>
          <input placeholder="Họ và tên" className="w-full border rounded px-3 py-2" required />
          <input placeholder="Số điện thoại" className="w-full border rounded px-3 py-2" required />
          <textarea placeholder="Nội dung" rows={3} className="w-full border rounded px-3 py-2" />
          <button className="bg-emerald-600 text-white px-4 py-2 rounded">Gửi</button>
        </form>
      </div>
    </div>
  );
}
