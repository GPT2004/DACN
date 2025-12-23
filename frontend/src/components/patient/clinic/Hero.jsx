/* eslint-disable no-console */
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../../context/AuthContext';

export default function Hero({ onBook }) {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();

  const handleBookClick = () => {
    console.log('Book button clicked, isAuthenticated:', isAuthenticated);
    if (!isAuthenticated) {
      navigate(`/login?returnUrl=${encodeURIComponent('/patient/appointments/book')}`);
    } else {
      navigate('/patient/appointments/book');
    }
  };

  const handleSpecialtyClick = (specialty) => {
    navigate(`/specialty/${encodeURIComponent(specialty)}`);
  };
  const specialtiesInfo = [
    {
      icon: '🏥',
      title: 'Nội tổng quát',
      desc: 'Khám chữa các bệnh về hệ thống nội tạng, chuẩn đoán và điều trị bệnh lý nội khoa'
    },
    {
      icon: '❤️',
      title: 'Tim mạch',
      desc: 'Chẩn đoán và điều trị các bệnh về tim, mạch máu và huyết áp'
    },
    {
      icon: '🍽️',
      title: 'Tiêu hóa',
      desc: 'Chuyên môn về bệnh đường tiêu hóa, dạ dày, gan và tụy'
    },
    {
      icon: '⚡',
      title: 'Nội tiết',
      desc: 'Điều trị các rối loạn về nội tiết tố, đái tháo đường và béo phì'
    },
    {
      icon: '🧴',
      title: 'Da liễu',
      desc: 'Chữa các bệnh về da, dị ứng da và thẩm mỹ da'
    },
    {
      icon: '👂',
      title: 'Tai Mũi Họng',
      desc: 'Khám chữa bệnh tai, mũi, họng và các vấn đề NAG'
    },
    {
      icon: '💨',
      title: 'Hô hấp',
      desc: 'Điều trị các bệnh về hệ hô hấp, hen suyễn và viêm phổi'
    }
  ];

  return (
    <section className="bg-gradient-to-r from-emerald-600 to-teal-500 text-white pt-20 pb-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-12">
          <h1 className="text-5xl font-extrabold mb-6">Phòng Khám Tư Nhân</h1>
          <p className="text-xl mb-6 opacity-95">Trung tâm chăm sóc sức khỏe toàn diện với đội ngũ bác sĩ chuyên khoa, trang thiết bị hiện đại và dịch vụ khám chữa bệnh tận tâm, chuyên nghiệp.</p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div className="bg-white bg-opacity-15 rounded-lg p-4 backdrop-blur">
              <div className="text-3xl mb-2">👨‍⚕️</div>
              <div className="font-semibold">Đội ngũ bác sĩ</div>
              <div className="text-sm opacity-90">Bác sĩ chuyên khoa giàu kinh nghiệm</div>
            </div>
            <div className="bg-white bg-opacity-15 rounded-lg p-4 backdrop-blur">
              <div className="text-3xl mb-2">⏱️</div>
              <div className="font-semibold">Đặt lịch trực tuyến</div>
              <div className="text-sm opacity-90">Dễ dàng và nhanh chóng 24/7</div>
            </div>
            <div className="bg-white bg-opacity-15 rounded-lg p-4 backdrop-blur">
              <div className="text-3xl mb-2">💊</div>
              <div className="font-semibold">Dịch vụ toàn diện</div>
              <div className="text-sm opacity-90">Khám, xét nghiệm, kê đơn tại một nơi</div>
            </div>
            <div className="bg-white bg-opacity-15 rounded-lg p-4 backdrop-blur">
              <div className="text-3xl mb-2">🏆</div>
              <div className="font-semibold">Chất lượng hàng đầu</div>
              <div className="text-sm opacity-90">Sự hài lòng của bệnh nhân là ưu tiên</div>
            </div>
          </div>

          <div className="flex gap-4 flex-wrap">
            <button
              onClick={handleBookClick}
              className="bg-white text-emerald-600 font-semibold px-6 py-3 rounded-lg shadow-lg hover:shadow-xl hover:scale-105 transition transform"
            >
              Đặt lịch khám ngay
            </button>
            <a href="#services" className="bg-white bg-opacity-20 text-white px-6 py-3 rounded-lg hover:bg-opacity-30 transition border border-white border-opacity-50">
              Tìm hiểu chuyên khoa →
            </a>
          </div>
        </div>

        <div className="mt-12 pt-12 border-t border-white border-opacity-20">
          <h2 className="text-3xl font-bold mb-8">7 Chuyên Khoa Phục Vụ</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {specialtiesInfo.map((specialty, index) => (
              <button
                key={index}
                onClick={() => handleSpecialtyClick(specialty.title)}
                className="bg-white bg-opacity-10 rounded-lg p-5 backdrop-blur hover:bg-opacity-20 hover:scale-105 transition transform text-left cursor-pointer"
              >
                <div className="text-4xl mb-3">{specialty.icon}</div>
                <div className="font-semibold text-lg mb-2">{specialty.title}</div>
                <p className="text-sm opacity-90 leading-relaxed">{specialty.desc}</p>
                <div className="mt-3 text-sm opacity-75 flex items-center gap-1">
                  Xem chi tiết →
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
