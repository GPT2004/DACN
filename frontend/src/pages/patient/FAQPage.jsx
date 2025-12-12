import React, { useState } from 'react';
import PublicHeader from '../../components/common/PublicHeader';
import PatientPageHeader from '../../components/patient/PatientPageHeader';
import Footer from '../../components/patient/clinic/Footer';
import { ChevronDown, HelpCircle } from 'lucide-react';

export default function FAQPage() {
  const [expandedId, setExpandedId] = useState(null);

  const faqs = [
    {
      id: 1,
      question: 'Làm sao để đặt lịch hẹn?',
      answer: 'Bạn có thể đặt lịch hẹn trực tiếp trên website hoặc gọi điện cho phòng khám để được tư vấn.'
    },
    {
      id: 2,
      question: 'Phí khám bao nhiêu?',
      answer: 'Phí khám tùy thuộc vào chuyên khoa. Vui lòng liên hệ phòng khám để được biết chi tiết.'
    },
    {
      id: 3,
      question: 'Cần chuẩn bị gì khi đi khám?',
      answer: 'Hãy mang theo CMND/CCCD, thẻ bảo hiểm (nếu có), và các đơn thuốc cũ của bạn.'
    },
    {
      id: 4,
      question: 'Phòng khám có tiêm chủng không?',
      answer: 'Có, phòng khám cung cấp dịch vụ tiêm chủng. Vui lòng liên hệ để đặt lịch.'
    },
    {
      id: 5,
      question: 'Giờ làm việc của phòng khám?',
      answer: 'Phòng khám hoạt động từ 7:30 - 17:00 từ thứ 2 đến thứ 7. Chủ nhật nghỉ.'
    },
  ];

  return (
    <div className="flex flex-col min-h-screen">
      <PublicHeader />
      <PatientPageHeader title="❓ Câu hỏi thường gặp" description="Những câu hỏi thường gặp về phòng khám Minh An" />
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">

        <div className="space-y-4">
        {faqs.map((faq) => (
          <div
            key={faq.id}
            className="bg-white rounded-lg shadow-md overflow-hidden"
          >
            <button
              onClick={() => setExpandedId(expandedId === faq.id ? null : faq.id)}
              className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition"
            >
              <div className="flex items-center gap-4 text-left">
                <HelpCircle className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                <span className="font-semibold text-gray-900">{faq.question}</span>
              </div>
              <ChevronDown
                className={`w-5 h-5 text-gray-600 transition-transform ${
                  expandedId === faq.id ? 'rotate-180' : ''
                }`}
              />
            </button>

            {expandedId === faq.id && (
              <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
                <p className="text-gray-700">{faq.answer}</p>
              </div>
            )}
          </div>
        ))}
        </div>
      </main>
      <Footer />
    </div>
  );
}
