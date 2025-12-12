import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getAllDoctorsPublic } from '../services/doctorService';
import { useAuth } from '../context/AuthContext';
import PublicHeader from '../components/common/PublicHeader';
import { ChevronLeft, Award, Users, Zap, Shield } from 'lucide-react';

// Specialty descriptions
const specialtyInfo = {
  'Nội tổng quát': {
    description: 'Khoa Nội tổng quát là chuyên khoa cơ bản khám chữa các bệnh về hệ thống nội tạng. Đội ngũ bác sĩ chuyên môn cao sẽ giúp bạn chuẩn đoán chính xác và điều trị hiệu quả các bệnh lý nội khoa từ đơn giản đến phức tạp. Chúng tôi cung cấp dịch vụ khám tổng quát, xét nghiệm máu toàn diện, siêu âm bụng và tư vấn sức khỏe.', 
    icon: '🏥',
    services: [
      'Khám sức khỏe tổng quát',
      'Chẩn đoán bệnh lý nội tạng',
      'Xét nghiệm máu và nước tiểu',
      'Siêu âm bụng và các cơ quan',
      'Quản lý bệnh lý mãn tính',
      'Tư vấn phòng ngừa bệnh tật',
      'Điều trị các bệnh nhiễm trùng',
    ],
  },
  'Tim mạch': {
    description: 'Khoa Tim mạch chuyên chẩn đoán và điều trị các bệnh về tim, mạch máu, huyết áp và các vấn đề tim mạch khác. Bác sĩ tim mạch giàu kinh nghiệm sử dụng các kỹ thuật chẩn đoán hiện đại như siêu âm tim, điện tâm đồ, và các xét nghiệm tim mạch chuyên biệt. Chúng tôi cung cấp dịch vụ khám tim mạch toàn diện, phòng ngừa bệnh tim, quản lý huyết áp cao, và điều trị các bệnh tim mạch mãn tính.',
    icon: '❤️',
    services: [
      'Khám tim mạch và huyết áp',
      'Siêu âm tim (Echo)',
      'Điện tâm đồ (ECG)',
      'Điều trị tăng huyết áp',
      'Quản lý bệnh tim mạch',
      'Phòng ngừa tai biến',
      'Tư vấn chế độ sống lành mạnh',
    ],
  },
  'Tiêu hóa': {
    description: 'Khoa Tiêu hóa chuyên chẩn đoán và điều trị các bệnh về đường tiêu hóa, dạ dày, gan, tụy và ruột. Bác sĩ tiêu hóa có kinh nghiệm trong điều trị viêm dạ dày, loét dạ dày, trào ngược axit, viêm gan, và các bệnh lý đường tiêu hóa phức tạp. Chúng tôi sử dụng các phương pháp chẩn đoán hiện đại và điều trị tiên tiến để mang lại sức khỏe tốt nhất cho bệnh nhân.',
    icon: '🍽️',
    services: [
      'Khám tiêu hóa và tư vấn chế độ ăn',
      'Nội soi dạ dày không đau',
      'Nội soi đại tràng',
      'Siêu âm gan, tụy',
      'Xét nghiệm chức năng gan',
      'Điều trị viêm dạ dày và loét',
      'Quản lý bệnh lý gan',
    ],
  },
  'Nội tiết': {
    description: 'Khoa Nội tiết chuyên điều trị các rối loạn hormon và tuyến nội tiết: đái tháo đường, bệnh tuyến giáp, béo phì, và các bệnh lý nội tiết khác. Bác sĩ nội tiết giàu kinh nghiệm trong chẩn đoán sớm và điều trị hiệu quả các bệnh nội tiết. Chúng tôi cung cấp dịch vụ quản lý đái tháo đường toàn diện, điều trị tuyến giáp, và tư vấn chế độ ăn uống phù hợp.',
    icon: '⚡',
    services: [
      'Khám nội tiết và xét nghiệm hormon',
      'Điều trị đái tháo đường',
      'Quản lý bệnh tuyến giáp',
      'Điều trị béo phì',
      'Tư vấn chế độ ăn và tập luyện',
      'Siêu âm tuyến giáp',
      'Theo dõi mức đường huyết',
    ],
  },
  'Da liễu': {
    description: 'Khoa Da liễu chuyên điều trị các bệnh về da: mụn trứng cá, eczema, viêm da dị ứng, nấm da, và các bệnh da khác. Bác sĩ da liễu sử dụng các phương pháp điều trị an toàn và hiệu quả, từ điều trị ngoài da đến điều trị nội khoa. Chúng tôi cung cấp dịch vụ chẩn đoán chính xác, điều trị toàn diện, và tư vấn chăm sóc da dài hạn.',
    icon: '🧴',
    services: [
      'Khám da và chẩn đoán bệnh da',
      'Điều trị mụn trứng cá',
      'Quản lý eczema và viêm da dị ứng',
      'Điều trị nấm da và nhiễm trùng',
      'Laser trị liệu da',
      'Tư vấn chăm sóc da',
      'Điều trị sẹo và sắc tố',
    ],
  },
  'Tai Mũi Họng': {
    description: 'Khoa Tai Mũi Họng chuyên khám và điều trị các bệnh về tai, mũi, họng, thanh quản và các vấn đề tai mũi họng khác. Bác sĩ tai mũi họng giàu kinh nghiệm trong điều trị viêm mũi xoang, viêm họng, viêm tai giữa, và các bệnh tai mũi họng phức tạp. Chúng tôi sử dụng các kỹ thuật chẩn đoán hiện đại và các phương pháp điều trị tiên tiến.',
    icon: '👂',
    services: [
      'Khám tai mũi họng toàn diện',
      'Điều trị viêm mũi xoang',
      'Quản lý viêm họng và đau họng',
      'Điều trị viêm tai giữa',
      'Kiểm tra thính lực',
      'Nội soi mũi và họng',
      'Tư vấn và phòng ngừa',
    ],
  },
  'Hô hấp': {
    description: 'Khoa Hô hấp chuyên chẩn đoán và điều trị các bệnh về phổi, đường hô hấp, và các vấn đề hô hấp khác. Bác sĩ hô hấp có kinh nghiệm trong điều trị hen phế quản, viêm phổi, bệnh phổi tắc nghẽn mãn tính (COPD), lao phổi, và các bệnh phổi khác. Chúng tôi cung cấp dịch vụ kiểm tra chức năng phổi, chẩn đoán bệnh hô hấp, và điều trị toàn diện.',
    icon: '💨',
    services: [
      'Khám hô hấp và chẩn đoán bệnh phổi',
      'Thử chức năng phổi',
      'Quản lý hen phế quản',
      'Điều trị viêm phổi',
      'Chụp X-quang ngực',
      'CT phổi và các xét nghiệm chuyên biệt',
      'Tư vấn phòng ngừa bệnh phổi',
    ],
  },
};

export default function SpecialtyDetail() {
  const { specialty } = useParams();
  const navigate = useNavigate();
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);

  const decodedSpecialty = decodeURIComponent(specialty || '');
  const info = specialtyInfo[decodedSpecialty] || {
    description: 'Chuyên khoa Y tế chuyên môn cao.',
    icon: '🏥',
    services: [],
  };
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        setLoading(true);
        const res = await getAllDoctorsPublic({ specialty: decodedSpecialty, limit: 50 });
        if (!mounted) return;
        setDoctors(res.data.doctors || res.data || []);
      } catch (err) {
        console.error('Load doctors error', err);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => { mounted = false; };
  }, [decodedSpecialty]);

  return (
    <div className="min-h-screen bg-gray-50">
      <PublicHeader />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-8 font-semibold transition hover:translate-x-1"
        >
          <ChevronLeft size={20} />
          Quay lại trang chủ
        </button>

        {/* Hero Section */}
        <div className="bg-gradient-to-r from-emerald-600 to-teal-500 text-white rounded-2xl shadow-2xl p-8 mb-8 animate-fade-in">
          <div className="flex items-start gap-6">
            <div className="text-7xl animate-bounce" style={{ animationDelay: '0.2s' }}>
              {info.icon}
            </div>
            <div className="flex-1">
              <h1 className="text-5xl font-extrabold mb-4">{decodedSpecialty}</h1>
              <p className="text-lg opacity-95 leading-relaxed">{info.description}</p>
            </div>
          </div>
        </div>

        {/* Services Section */}
        {info.services && info.services.length > 0 && (
          <div className="bg-white rounded-2xl shadow-lg p-8 mb-8 hover:shadow-2xl transition-shadow duration-300">
            <div className="flex items-center gap-3 mb-6">
              <Zap className="text-emerald-600" size={28} />
              <h2 className="text-3xl font-bold">Dịch Vụ Cung Cấp</h2>
            </div>
            <p className="text-gray-600 mb-6">Các dịch vụ chuyên ngành mà phòng khám chúng tôi cung cấp:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-4">
              {info.services.map((svc, index) => (
                <div 
                  key={svc} 
                  className="flex items-start gap-3 p-4 bg-gradient-to-r from-emerald-50 to-teal-50 rounded-lg border-l-4 border-emerald-600 hover:shadow-md transform hover:translate-x-1 transition-all duration-300"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="w-6 h-6 bg-emerald-600 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0 mt-1">
                    {index + 1}
                  </div>
                  <span className="text-gray-800 font-medium">{svc}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Why Choose Us Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-2xl transition-shadow duration-300 group">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-emerald-100 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                <Award className="text-emerald-600" size={24} />
              </div>
              <h3 className="text-xl font-bold">Chuyên Môn Cao</h3>
            </div>
            <p className="text-gray-600">Đội ngũ bác sĩ có trình độ chuyên khoa cao, kinh nghiệm lâu năm trong lĩnh vực chuyên khoa của mình. Tất cả bác sĩ đều có bằng cấp chuyên ngành được công nhận.</p>
          </div>

          <div className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-2xl transition-shadow duration-300 group">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                <Shield className="text-blue-600" size={24} />
              </div>
              <h3 className="text-xl font-bold">Trang Thiết Bị Hiện Đại</h3>
            </div>
            <p className="text-gray-600">Sử dụng các máy móc, thiết bị chẩn đoán và điều trị hiện đại nhất. Đảm bảo chẩn đoán chính xác và điều trị hiệu quả nhất cho bệnh nhân.</p>
          </div>

          <div className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-2xl transition-shadow duration-300 group">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                <Users className="text-purple-600" size={24} />
              </div>
              <h3 className="text-xl font-bold">Dịch Vụ Tận Tâm</h3>
            </div>
            <p className="text-gray-600">Đội ngũ nhân viên y tế lịch sự, tận tâm. Luôn sẵn sàng lắng nghe, tư vấn và giúp đỡ bệnh nhân trong quá trình khám chữa bệnh.</p>
          </div>

          <div className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-2xl transition-shadow duration-300 group">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                <Zap className="text-orange-600" size={24} />
              </div>
              <h3 className="text-xl font-bold">Xử Lý Nhanh Chóng</h3>
            </div>
            <p className="text-gray-600">Thời gian chờ đợi ngắn, quy trình khám chữa bệnh hiệu quả. Bệnh nhân có thể nhận được kết quả xét nghiệm nhanh chóng.</p>
          </div>
        </div>

        {/* FAQ Section */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
          <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
            <span className="text-2xl">❓</span> Câu Hỏi Thường Gặp
          </h2>
          <div className="space-y-4">
            <div className="border-l-4 border-emerald-600 pl-4 py-2 hover:bg-gray-50 rounded transition">
              <h4 className="font-semibold text-lg mb-2">Tôi cần phải làm gì trước khi khám?</h4>
              <p className="text-gray-600">Vui lòng đặt lịch hẹn trước để có thể xếp lịch khám hợp lý. Chuẩn bị các giấy tờ y tế liên quan nếu có.</p>
            </div>
            <div className="border-l-4 border-emerald-600 pl-4 py-2 hover:bg-gray-50 rounded transition">
              <h4 className="font-semibold text-lg mb-2">Thời gian khám bao lâu?</h4>
              <p className="text-gray-600">Thời gian khám thường kéo dài từ 15-30 phút, tùy vào tình trạng bệnh của bệnh nhân.</p>
            </div>
            <div className="border-l-4 border-emerald-600 pl-4 py-2 hover:bg-gray-50 rounded transition">
              <h4 className="font-semibold text-lg mb-2">Chi phí khám bao nhiêu?</h4>
              <p className="text-gray-600">Chi phí khám thay đổi tùy theo dịch vụ và xét nghiệm. Vui lòng liên hệ để biết chi tiết.</p>
            </div>
            <div className="border-l-4 border-emerald-600 pl-4 py-2 hover:bg-gray-50 rounded transition">
              <h4 className="font-semibold text-lg mb-2">Có thể tư vấn trực tuyến không?</h4>
              <p className="text-gray-600">Có, chúng tôi cung cấp dịch vụ tư vấn trực tuyến. Vui lòng liên hệ để đặt lịch.</p>
            </div>
          </div>
        </div>

        {/* Doctors Section */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <h2 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Users className="text-emerald-600" size={32} />
            Bác Sĩ Chuyên Khoa
          </h2>
          <p className="text-gray-600 mb-8">Đội ngũ bác sĩ giàu kinh nghiệm, tâm huyết với nghề, luôn sẵn sàng phục vụ bạn</p>
          
          {loading && (
            <div className="flex justify-center py-12">
              <div className="animate-spin">
                <div className="w-12 h-12 border-4 border-emerald-600 border-t-transparent rounded-full" />
              </div>
            </div>
          )}
          
          {!loading && doctors.length === 0 && (
            <div className="text-center py-12 text-gray-500">Chưa có bác sĩ nào trong chuyên khoa này.</div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {doctors.map((doctor, idx) => {
              const id = doctor.id || doctor.userId || (doctor.user && doctor.user.id) || doctor._id;
              const user = doctor.user || {};
              const avatar = doctor.avatar_url || user.avatar_url || user.avatar || doctor.photo || doctor.profilePicture || null;
              const name = user.full_name || doctor.fullName || doctor.name || user.name || 'Bác sĩ';
              const specialtiesText = doctor.specialty || (doctor.specialties && doctor.specialties.join(', ')) || user.specialties?.join(', ') || decodedSpecialty;
              const bio = doctor.bio || doctor.description || user.bio || 'Bác sĩ chuyên khoa';

              const initials = name ? name.split(' ').map(n => n[0]).slice(0,2).join('') : 'BS';

              return (
                <div 
                  key={id || Math.random()} 
                  className="bg-gradient-to-br from-white to-gray-50 border-2 border-gray-200 rounded-xl overflow-hidden hover:shadow-2xl hover:border-emerald-400 transition-all duration-300 transform hover:-translate-y-2 group"
                  style={{ animationDelay: `${idx * 0.1}s` }}
                >
                  {/* Doctor Avatar */}
                  <div className="relative h-32 bg-gradient-to-r from-emerald-600 to-teal-500 flex items-end justify-center pb-4">
                    <div className="w-28 h-28 rounded-full flex items-center justify-center text-4xl overflow-hidden bg-white shadow-lg border-4 border-white group-hover:scale-110 transition-transform duration-300">
                      {avatar ? (
                        <img src={avatar} alt={name} className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-2xl text-emerald-600 font-bold">{initials}</div>
                      )}
                    </div>
                  </div>

                  {/* Doctor Info */}
                  <div className="px-4 py-6 text-center">
                    <div className="font-bold text-xl mb-1 text-gray-800">{name}</div>
                    <div className="text-sm text-emerald-600 font-semibold mb-3">{specialtiesText}</div>
                    <p className="text-sm text-gray-600 mb-6 h-12 line-clamp-2">{bio}</p>
                    
                    {/* Rating */}
                    <div className="flex items-center justify-center gap-1 mb-6">
                      <span className="text-yellow-400">★★★★★</span>
                      <span className="text-sm text-gray-600">(5.0)</span>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-2 justify-center">
                      <button
                        onClick={() => navigate(`/doctors/${id}`)}
                        className="flex-1 px-3 py-2 bg-emerald-600 text-white rounded-lg font-semibold hover:bg-emerald-700 transition-all duration-200 transform hover:scale-105"
                      >
                        Xem chi tiết
                      </button>
                      <button
                        onClick={() => {
                          if (!isAuthenticated) {
                            const currentPath = window.location.pathname;
                            return navigate(`/login?returnUrl=${encodeURIComponent(currentPath)}`);
                          }
                          return navigate(`/doctors/${id}`);
                        }}
                        className="flex-1 px-3 py-2 border-2 border-emerald-600 text-emerald-600 font-semibold rounded-lg hover:bg-emerald-50 transition-all duration-200 transform hover:scale-105"
                      >
                        Đặt lịch
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.6s ease-out;
        }
      `}</style>
    </div>
  );
}
