import React, { useState } from 'react';
import PublicHeader from '../../components/common/PublicHeader';
import PatientPageHeader from '../../components/patient/PatientPageHeader';
import Footer from '../../components/patient/clinic/Footer';
import { Send, MessageCircle } from 'lucide-react';

export default function MessagesPage() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'received',
      sender: 'BS. Nguyễn Văn A',
      content: 'Kết quả xét nghiệm của bạn đã sẵn sàng. Vui lòng quay lại phòng khám để lấy kết quả.',
      timestamp: '14:30 - 15/11/2025'
    },
    {
      id: 2,
      type: 'sent',
      sender: 'Bạn',
      content: 'Cảm ơn bác sĩ! Tôi sẽ đến lấy kết quả vào chiều mai.',
      timestamp: '15:45 - 15/11/2025'
    },
  ]);

  const [newMessage, setNewMessage] = useState('');

  const handleSendMessage = () => {
    if (newMessage.trim()) {
      setMessages([...messages, {
        id: messages.length + 1,
        type: 'sent',
        sender: 'Bạn',
        content: newMessage,
        timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }) + ' - ' + new Date().toLocaleDateString('vi-VN')
      }]);
      setNewMessage('');
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <PublicHeader />
      <PatientPageHeader title="💬 Tin nhắn" description="Trò chuyện với bác sĩ và nhân viên phòng khám" />
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">

        <div className="bg-white rounded-lg shadow-md overflow-hidden" style={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <MessageCircle className="w-12 h-12 mx-auto mb-4 text-gray-300" />
              <p>Chưa có tin nhắn nào</p>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.type === 'sent' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    msg.type === 'sent'
                      ? 'bg-emerald-500 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  {msg.type === 'received' && (
                    <p className="text-sm font-semibold mb-1">{msg.sender}</p>
                  )}
                  <p>{msg.content}</p>
                  <p className={`text-xs mt-1 ${
                    msg.type === 'sent' ? 'text-emerald-100' : 'text-gray-500'
                  }`}>
                    {msg.timestamp}
                  </p>
                </div>
              </div>
            ))
          )}
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 p-4 bg-gray-50">
            <div className="flex gap-3">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Nhập tin nhắn..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            />
            <button
              onClick={handleSendMessage}
              className="bg-emerald-600 text-white px-6 py-2 rounded-lg hover:bg-emerald-700 transition flex items-center gap-2"
            >
              <Send size={18} />
              Gửi
            </button>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
