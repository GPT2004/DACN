/* eslint-disable no-unused-vars */
import React, { useState, useEffect } from 'react';
import PublicHeader from '../components/common/PublicHeader';
import Hero from '../components/patient/clinic/Hero';
import Services from '../components/patient/clinic/Services';
import Doctors from '../components/patient/clinic/Doctors';
import Contact from '../components/patient/clinic/Contact';
import Footer from '../components/patient/clinic/Footer';
import AIAssistant from '../components/patient/AIAssistant';
import { useNavigate } from 'react-router-dom';
import { ArrowDown, Award, Clock, Users, HeartHandshake } from 'lucide-react';

export default function HomePage() {
  const navigate = useNavigate();
  const [selectedSpecialty, setSelectedSpecialty] = useState(null);
  const [scrollPosition, setScrollPosition] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollPosition(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleSelectSpecialty = (spec) => {
    setSelectedSpecialty(spec);
    setTimeout(() => {
      const section = document.getElementById('doctors-section');
      if (section) section.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  const handleBookNow = () => {
    navigate('/login?returnUrl=/');
  };

  const stats = [
    { icon: Users, label: 'Bác sĩ chuyên khoa', value: '14+', color: 'emerald' },
    { icon: Clock, label: 'Năm kinh nghiệm', value: '10+', color: 'blue' },
    { icon: Award, label: 'Chuyên khoa', value: '7', color: 'purple' },
    { icon: HeartHandshake, label: 'Bệnh nhân hài lòng', value: '5000+', color: 'pink' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <PublicHeader />
      
      {/* Hero Section */}
      <Hero onBook={handleBookNow} />
      
      {/* Scroll Indicator */}
      <div className="flex justify-center pb-8 animate-bounce">
        <ArrowDown className="text-emerald-600" size={24} />
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Stats Section */}
        <div className="mb-16 -mt-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              const colorClass = {
                emerald: 'from-emerald-600 to-teal-500',
                blue: 'from-blue-600 to-cyan-500',
                purple: 'from-purple-600 to-pink-500',
                pink: 'from-pink-600 to-rose-500',
              }[stat.color];

              return (
                <div
                  key={index}
                  className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-lg p-6 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-gray-100"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className={`bg-gradient-to-br ${colorClass} rounded-xl p-4 w-16 h-16 flex items-center justify-center mb-4 text-white shadow-lg`}>
                    <Icon size={28} />
                  </div>
                  <div className="text-3xl font-bold text-gray-800 mb-2">{stat.value}</div>
                  <p className="text-gray-600 font-medium">{stat.label}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Services Section */}
        <section className="mb-16">
          <div className="mb-8">
            <h2 className="text-4xl font-bold text-center mb-4 text-gray-800">
              Khám Phá Dịch Vụ Của Chúng Tôi
            </h2>
            <p className="text-center text-gray-600 max-w-2xl mx-auto">
              Phòng khám cung cấp dịch vụ y tế chuyên ngành đầy đủ với đội ngũ bác sĩ giàu kinh nghiệm
            </p>
          </div>
          <div className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-2xl transition-all duration-300">
            <Services onSelectSpecialty={handleSelectSpecialty} />
          </div>
        </section>

        {/* Featured Doctors Section */}
        <section id="doctors-section" className="mb-16 scroll-mt-20">
          <div className="mb-8">
            <h2 className="text-4xl font-bold text-center mb-4 text-gray-800">
              Bác Sĩ Của Chúng Tôi
            </h2>
            <p className="text-center text-gray-600 max-w-2xl mx-auto">
              Đội ngũ bác sĩ chuyên khoa được đào tạo bài bản, nhiều năm kinh nghiệm lâm sàng
            </p>
          </div>
          <div className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-2xl transition-all duration-300">
            <Doctors specialty={selectedSpecialty} />
          </div>
        </section>

        {/* Why Choose Us Section */}
        <section className="mb-16 bg-gradient-to-r from-emerald-600 to-teal-500 text-white rounded-2xl shadow-2xl p-12">
          <h2 className="text-4xl font-bold text-center mb-12">Tại Sao Chọn Chúng Tôi?</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: '🏥',
                title: 'Cơ Sở Hiện Đại',
                desc: 'Trang thiết bị y tế tân tiến, phòng khám sạch sẽ và quy chuẩn'
              },
              {
                icon: '👨‍⚕️',
                title: 'Bác Sĩ Giỏi',
                desc: 'Đội ngũ bác sĩ chuyên khoa, giàu kinh nghiệm và tâm huyết'
              },
              {
                icon: '⏰',
                title: 'Dịch Vụ 24/7',
                desc: 'Phục vụ bệnh nhân 24 giờ, 7 ngày trong tuần'
              },
              {
                icon: '💰',
                title: 'Giá Cả Hợp Lý',
                desc: 'Giá khám chữa bệnh hợp lý, có gói khám ưu đãi'
              },
            ].map((item, index) => (
              <div
                key={index}
                className="bg-white bg-opacity-10 backdrop-blur rounded-xl p-6 hover:bg-opacity-20 transition-all duration-300 transform hover:scale-105 border border-white border-opacity-20"
              >
                <div className="text-4xl mb-4">{item.icon}</div>
                <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                <p className="text-white text-opacity-90">{item.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Contact Section */}
        <section id="contact" className="mb-16">
          <div className="mb-8">
            <h2 className="text-4xl font-bold text-center mb-4 text-gray-800">
              Liên Hệ Với Chúng Tôi
            </h2>
            <p className="text-center text-gray-600 max-w-2xl mx-auto">
              Hãy liên hệ nếu bạn có bất kỳ câu hỏi hoặc muốn đặt lịch khám
            </p>
          </div>
          <div className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-2xl transition-all duration-300">
            <Contact />
          </div>
        </section>

        {/* CTA Section */}
        <section className="mb-16 bg-gradient-to-r from-blue-600 to-emerald-600 text-white rounded-2xl shadow-2xl p-12 text-center transform hover:scale-105 transition-transform duration-300">
          <h2 className="text-4xl font-bold mb-4">Sẵn Sàng Khám Chữa Bệnh?</h2>
          <p className="text-lg opacity-90 mb-8 max-w-2xl mx-auto">
            Đặt lịch khám trực tuyến ngay hôm nay và nhận tư vấn từ bác sĩ chuyên khoa
          </p>
          <button
            onClick={handleBookNow}
            className="bg-white text-emerald-600 font-bold px-8 py-4 rounded-lg hover:bg-gray-100 transition-all duration-200 transform hover:scale-110 shadow-lg"
          >
            Đặt Lịch Ngay
          </button>
        </section>
      </main>

      <Footer />
      
      {/* AI Assistant */}
      <AIAssistant />

      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes slideInLeft {
          from {
            opacity: 0;
            transform: translateX(-30px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        section {
          animation: fadeInUp 0.6s ease-out forwards;
        }
      `}</style>
    </div>
  );
}
