import React, { useState } from 'react';
import { Bot, X, Send, Sparkles, Loader2, Activity, Heart } from 'lucide-react';
import axios from 'axios';

const AI_COMBINED_URL = process.env.REACT_APP_AI_COMBINED_SERVICE_URL || 'http://localhost:5000';

export default function DoctorAIDiagnosis() {
  const [isOpen, setIsOpen] = useState(false);
  const [formData, setFormData] = useState({
    // Heart data
    Age: '',
    Sex: 'M',
    ChestPainType: 'ATA',
    RestingBP: '',
    Cholesterol: '',
    FastingBS: '0',
    RestingECG: 'Normal',
    MaxHR: '',
    ExerciseAngina: 'N',
    Oldpeak: '',
    ST_Slope: 'Up',
    // Diabetes data
    gender: 'male',
    age: '',
    hypertension: '0',
    heart_disease: '0',
    smoking_history: 'never',
    bmi: '',
    HbA1c_level: '',
    blood_glucose_level: ''
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      // Send as form data to Flask
      const formDataToSend = new URLSearchParams();
      Object.keys(formData).forEach(key => {
        if (formData[key]) {
          formDataToSend.append(key, formData[key]);
        }
      });

      const response = await axios.post(`${AI_COMBINED_URL}/api/predict`, formDataToSend.toString(), {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });

      setResult(response.data);
    } catch (err) {
      console.error('AI Diagnosis Error:', err);
      setError('Không thể kết nối tới AI service. Vui lòng kiểm tra Python server.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (prob) => {
    if (prob < 0.3) return 'text-green-600 bg-green-50';
    if (prob < 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <>
      {/* Trigger Button in Sidebar */}
      <button
        onClick={() => setIsOpen(true)}
        className="w-full flex items-center gap-3 px-4 py-3 rounded-md transition text-purple-700 hover:bg-purple-50 border-2 border-purple-200"
      >
        <Activity size={20} />
        <span className="font-semibold">AI Chuẩn đoán</span>
        <Sparkles size={16} className="ml-auto" />
      </button>

      {/* Modal */}
      {isOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] overflow-hidden animate-slideUp">
            {/* Header */}
            <div className="bg-gradient-to-r from-purple-600 to-indigo-600 p-6 text-white">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/20 rounded-xl backdrop-blur-sm">
                    <Activity size={28} />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold">AI Chuẩn Đoán Nguy Cơ Bệnh</h2>
                    <p className="text-purple-100 text-sm">Đánh giá nguy cơ tim mạch và tiểu đường</p>
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
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Heart Disease Section */}
                <div className="border-2 border-red-200 rounded-xl p-5 bg-red-50/30">
                  <h3 className="text-lg font-bold text-red-700 mb-4 flex items-center gap-2">
                    <Heart size={20} />
                    Dữ liệu Tim Mạch
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Tuổi</label>
                      <input type="number" name="Age" value={formData.Age} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg" required />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Giới tính</label>
                      <select name="Sex" value={formData.Sex} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg">
                        <option value="M">Nam</option>
                        <option value="F">Nữ</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Đau ngực</label>
                      <select name="ChestPainType" value={formData.ChestPainType} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg">
                        <option value="ATA">Điển hình</option>
                        <option value="NAP">Không điển hình</option>
                        <option value="ASY">Không triệu chứng</option>
                        <option value="TA">Atypical</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Huyết áp (mmHg)</label>
                      <input type="number" name="RestingBP" value={formData.RestingBP} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg" placeholder="VD: 138" />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Cholesterol (mg/dL)</label>
                      <input type="number" name="Cholesterol" value={formData.Cholesterol} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg" placeholder="VD: 214" />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Đường huyết đói</label>
                      <select name="FastingBS" value={formData.FastingBS} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg">
                        <option value="0">{'<120 mg/dL'}</option>
                        <option value="1">{'>120 mg/dL'}</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">ECG</label>
                      <select name="RestingECG" value={formData.RestingECG} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg">
                        <option value="Normal">Bình thường</option>
                        <option value="ST">ST thay đổi</option>
                        <option value="LVH">LVH</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Nhịp tim tối đa</label>
                      <input type="number" name="MaxHR" value={formData.MaxHR} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg" />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Đau khi gắng sức</label>
                      <select name="ExerciseAngina" value={formData.ExerciseAngina} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg">
                        <option value="N">Không</option>
                        <option value="Y">Có</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Oldpeak</label>
                      <input type="number" step="0.1" name="Oldpeak" value={formData.Oldpeak} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg" />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">ST Slope</label>
                      <select name="ST_Slope" value={formData.ST_Slope} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg">
                        <option value="Up">Lên</option>
                        <option value="Flat">Phẳng</option>
                        <option value="Down">Xuống</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Diabetes Section */}
                <div className="border-2 border-blue-200 rounded-xl p-5 bg-blue-50/30">
                  <h3 className="text-lg font-bold text-blue-700 mb-4 flex items-center gap-2">
                    <Activity size={20} />
                    Dữ liệu Tiểu Đường
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Tăng huyết áp</label>
                      <select name="hypertension" value={formData.hypertension} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg">
                        <option value="0">Không</option>
                        <option value="1">Có</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Bệnh tim</label>
                      <select name="heart_disease" value={formData.heart_disease} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg">
                        <option value="0">Không</option>
                        <option value="1">Có</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Hút thuốc</label>
                      <select name="smoking_history" value={formData.smoking_history} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg">
                        <option value="never">Chưa bao giờ</option>
                        <option value="former">Đã bỏ</option>
                        <option value="current">Đang hút</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">BMI</label>
                      <input type="number" step="0.1" name="bmi" value={formData.bmi} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg" />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">HbA1c (%)</label>
                      <input type="number" step="0.1" name="HbA1c_level" value={formData.HbA1c_level} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg" />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Đường huyết (mg/dL)</label>
                      <input type="number" name="blood_glucose_level" value={formData.blood_glucose_level} onChange={handleChange} className="w-full px-3 py-2 border rounded-lg" />
                    </div>
                  </div>
                </div>

                {error && (
                  <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl">
                    {error}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-3 px-6 rounded-xl hover:shadow-lg transition-all duration-300 disabled:opacity-50 flex items-center justify-center gap-2 font-medium text-lg"
                >
                  {loading ? (
                    <>
                      <Loader2 size={24} className="animate-spin" />
                      Đang phân tích...
                    </>
                  ) : (
                    <>
                      <Send size={24} />
                      Phân tích nguy cơ
                    </>
                  )}
                </button>
              </form>

              {/* Results */}
              {result && (
                <div className="mt-6 grid grid-cols-2 gap-4 animate-fadeIn">
                  {/* Heart Risk */}
                  <div className={`border-2 rounded-xl p-5 ${getRiskColor(result.cardio?.prob || 0)}`}>
                    <div className="flex items-center gap-3 mb-3">
                      <Heart size={24} />
                      <h4 className="font-bold text-lg">Nguy cơ Tim mạch</h4>
                    </div>
                    <div className="text-4xl font-bold mb-2">
                      {((result.cardio?.prob || 0) * 100).toFixed(1)}%
                    </div>
                    <div className="flex-1 bg-gray-200 rounded-full h-3 mb-2">
                      <div
                        className="bg-red-500 h-3 rounded-full transition-all duration-500"
                        style={{ width: `${(result.cardio?.prob || 0) * 100}%` }}
                      />
                    </div>
                    <p className="text-sm font-semibold">Mức độ: {result.cardio?.risk?.[0] || 'N/A'}</p>
                  </div>

                  {/* Diabetes Risk */}
                  <div className={`border-2 rounded-xl p-5 ${getRiskColor(result.diabetes?.prob || 0)}`}>
                    <div className="flex items-center gap-3 mb-3">
                      <Activity size={24} />
                      <h4 className="font-bold text-lg">Nguy cơ Tiểu đường</h4>
                    </div>
                    <div className="text-4xl font-bold mb-2">
                      {((result.diabetes?.prob || 0) * 100).toFixed(1)}%
                    </div>
                    <div className="flex-1 bg-gray-200 rounded-full h-3 mb-2">
                      <div
                        className="bg-blue-500 h-3 rounded-full transition-all duration-500"
                        style={{ width: `${(result.diabetes?.prob || 0) * 100}%` }}
                      />
                    </div>
                    <p className="text-sm font-semibold">Mức độ: {result.diabetes?.risk?.[0] || 'N/A'}</p>
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
