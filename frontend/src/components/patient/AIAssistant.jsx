import React, { useState } from 'react';
import { Bot, X, Send, Sparkles, Loader2 } from 'lucide-react';
import axios from 'axios';

const AI_SYMPTOM_URL = process.env.REACT_APP_AI_SYMPTOM_SERVICE_URL || 'http://localhost:5001';

export default function AIAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [symptoms, setSymptoms] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!symptoms.trim()) {
      setError('Vui lòng nhập triệu chứng');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post(`${AI_SYMPTOM_URL}/api/symptom`, {
        symptoms: symptoms.trim()
      });

      if (response.data.ok) {
        setResult(response.data);
      } else {
        setError('Không thể phân tích triệu chứng');
      }
    } catch (err) {
      console.error('AI Assistant Error:', err);
      setError('Không thể kết nối tới AI service. Vui lòng kiểm tra xem Python server có đang chạy không.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSymptoms('');
    setResult(null);
    setError('');
  };

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 bg-gradient-to-r from-emerald-500 to-teal-600 text-white p-4 rounded-full shadow-2xl hover:shadow-emerald-500/50 transition-all duration-300 hover:scale-110 z-50 group"
        aria-label="AI Trợ giúp"
        title="Mở AI Trợ Giúp Y Tế"
      >
        <Bot size={28} className="group-hover:rotate-12 transition-transform duration-300" />
        <span className="absolute -top-1 -right-1 flex h-5 w-5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-5 w-5 bg-emerald-500">
            <Sparkles size={12} className="m-auto text-white" />
          </span>
        </span>
      </button>

      {/* Modal */}
      {isOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden animate-slideUp">
            {/* Header */}
            <div className="bg-gradient-to-r from-emerald-500 to-teal-600 p-6 text-white">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/20 rounded-xl backdrop-blur-sm">
                    <Bot size={28} />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold">AI Trợ Giúp Y Tế</h2>
                    <p className="text-emerald-50 text-sm">Gợi ý khoa khám từ triệu chứng</p>
                  </div>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                >
                  <X size={24} />
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-160px)]">
              {/* Form */}
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Nhập triệu chứng của bạn
                  </label>
                  <textarea
                    value={symptoms}
                    onChange={(e) => setSymptoms(e.target.value)}
                    placeholder="Ví dụ: sốt, ho khan, đau họng, mệt mỏi..."
                    className="w-full min-h-[100px] px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
                    disabled={loading}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Nhập các triệu chứng, cách nhau bởi dấu phẩy
                  </p>
                </div>

                {error && (
                  <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl flex items-center gap-2">
                    <span className="text-sm">{error}</span>
                  </div>
                )}

                <div className="flex gap-3">
                  <button
                    type="submit"
                    disabled={loading}
                    className="flex-1 bg-gradient-to-r from-emerald-500 to-teal-600 text-white py-3 px-6 rounded-xl hover:shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
                  >
                    {loading ? (
                      <>
                        <Loader2 size={20} className="animate-spin" />
                        Đang phân tích...
                      </>
                    ) : (
                      <>
                        <Send size={20} />
                        Phân tích
                      </>
                    )}
                  </button>
                  {result && (
                    <button
                      type="button"
                      onClick={handleReset}
                      className="px-6 py-3 border border-gray-300 rounded-xl hover:bg-gray-50 transition-colors font-medium"
                    >
                      Làm mới
                    </button>
                  )}
                </div>
              </form>

              {/* Result */}
              {result && (
                <div className="mt-6 space-y-4 animate-fadeIn">
                  {/* Main Recommendation */}
                  <div className="bg-gradient-to-br from-emerald-50 to-teal-50 border-2 border-emerald-200 rounded-xl p-5">
                    <div className="flex items-start gap-3">
                      <div className="p-2 bg-emerald-500 rounded-lg">
                        <Sparkles size={20} className="text-white" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-bold text-lg text-gray-900 mb-1">
                          Khoa khám gợi ý
                        </h3>
                        <p className="text-2xl font-bold text-emerald-700 mb-2">
                          {result.suggested_department_vi}
                        </p>
                        {result.confidence && (
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-gray-600">Độ tin cậy:</span>
                            <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-xs">
                              <div
                                className="bg-emerald-500 h-2 rounded-full transition-all duration-500"
                                style={{ width: `${result.confidence * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-semibold text-emerald-600">
                              {(result.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* All Departments Probability */}
                  {result.topk && result.topk.length > 0 && (
                    <div className="bg-white border border-gray-200 rounded-xl p-5">
                      <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                        <span className="w-1 h-5 bg-emerald-500 rounded"></span>
                        Xác suất các khoa
                      </h4>
                      <div className="space-y-3">
                        {result.topk.slice(0, 5).map((item, idx) => (
                          <div key={idx} className="flex items-center gap-3">
                            <span className="text-sm font-medium text-gray-700 w-32">
                              {item.dept}
                            </span>
                            <div className="flex-1 bg-gray-100 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full transition-all duration-500 ${
                                  idx === 0 ? 'bg-emerald-500' : 'bg-gray-400'
                                }`}
                                style={{ width: `${item.prob * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-semibold text-gray-600 w-12 text-right">
                              {(item.prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Common Diseases */}
                  {result.diseases && result.diseases.length > 0 && (
                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-5">
                      <h4 className="font-semibold text-gray-900 mb-3">
                        Bệnh thường gặp (gợi ý)
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {result.diseases.map((disease, idx) => (
                          <span
                            key={idx}
                            className="px-3 py-1 bg-white border border-blue-200 rounded-full text-sm text-gray-700"
                          >
                            {disease}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Disclaimer */}
                  <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
                    <p className="text-xs text-yellow-800">
                      ⚠️ <strong>Lưu ý:</strong> Đây chỉ là gợi ý từ AI, không thay thế cho chẩn đoán y khoa. 
                      Vui lòng đặt lịch khám với bác sĩ để được tư vấn chính xác.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { 
            opacity: 0;
            transform: translateY(20px);
          }
          to { 
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
        .animate-slideUp {
          animation: slideUp 0.4s ease-out;
        }
      `}</style>
    </>
  );
}
