// frontend/src/components/patient/SymptomChecker.jsx
import React, { useState } from 'react';
import { Bot, AlertTriangle, CheckCircle, Info, Send, X, Plus } from 'lucide-react';
import { aiService } from '../../services/aiService';
import Button from '../common/Button';
import Badge from '../common/Badge';

export default function SymptomChecker() {
  const [symptoms, setSymptoms] = useState(['']);
  const [duration, setDuration] = useState('');
  const [severity, setSeverity] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showResult, setShowResult] = useState(false);

  const severityOptions = [
    { value: 'mild', label: 'Nhẹ', color: 'text-green-600' },
    { value: 'moderate', label: 'Trung bình', color: 'text-yellow-600' },
    { value: 'severe', label: 'Nghiêm trọng', color: 'text-red-600' },
  ];

  const handleAddSymptom = () => {
    setSymptoms([...symptoms, '']);
  };

  const handleRemoveSymptom = (index) => {
    setSymptoms(symptoms.filter((_, i) => i !== index));
  };

  const handleSymptomChange = (index, value) => {
    const newSymptoms = [...symptoms];
    newSymptoms[index] = value;
    setSymptoms(newSymptoms);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const validSymptoms = symptoms.filter(s => s.trim());
    
    if (validSymptoms.length === 0) {
      alert('Vui lòng nhập ít nhất một triệu chứng');
      return;
    }

    try {
      setLoading(true);
      const response = await aiService.checkSymptoms({
        symptoms: validSymptoms,
        duration,
        severity,
      });

      setResult(response.data);
      setShowResult(true);
    } catch (error) {
      console.error('Error checking symptoms:', error);
      alert('Có lỗi xảy ra. Vui lòng thử lại!');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSymptoms(['']);
    setDuration('');
    setSeverity('');
    setResult(null);
    setShowResult(false);
  };

  const getRiskColor = (level) => {
    const colors = {
      low: 'text-green-600 bg-green-50 border-green-200',
      medium: 'text-yellow-600 bg-yellow-50 border-yellow-200',
      high: 'text-red-600 bg-red-50 border-red-200',
    };
    return colors[level] || colors.low;
  };

  const getRiskIcon = (level) => {
    if (level === 'high') return <AlertTriangle className="w-5 h-5" />;
    if (level === 'medium') return <Info className="w-5 h-5" />;
    return <CheckCircle className="w-5 h-5" />;
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-t-lg p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 rounded-full bg-white bg-opacity-20 flex items-center justify-center">
            <Bot className="w-7 h-7" />
          </div>
          <div>
            <h2 className="text-2xl font-bold">AI Symptom Checker</h2>
            <p className="text-purple-100 text-sm">Kiểm tra triệu chứng với trí tuệ nhân tạo</p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-b-lg shadow-lg p-6">
        {/* Warning */}
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-yellow-800">
              <p className="font-semibold mb-1">Lưu ý quan trọng:</p>
              <ul className="list-disc list-inside space-y-1">
                <li>Công cụ này chỉ mang tính chất tham khảo, KHÔNG thay thế chẩn đoán của bác sĩ</li>
                <li>Nếu triệu chứng nghiêm trọng, vui lòng đến cơ sở y tế ngay lập tức</li>
                <li>Kết quả có thể không chính xác 100%</li>
              </ul>
            </div>
          </div>
        </div>

        {!showResult ? (
          /* Input Form */
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Symptoms */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Triệu chứng của bạn <span className="text-red-500">*</span>
              </label>
              <div className="space-y-3">
                {symptoms.map((symptom, index) => (
                  <div key={index} className="flex gap-2">
                    <input
                      type="text"
                      className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                      value={symptom}
                      onChange={(e) => handleSymptomChange(index, e.target.value)}
                      placeholder={`Triệu chứng ${index + 1} (VD: đau đầu, sốt cao, ho...)`}
                    />
                    {symptoms.length > 1 && (
                      <button
                        type="button"
                        onClick={() => handleRemoveSymptom(index)}
                        className="px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <button
                type="button"
                onClick={handleAddSymptom}
                className="mt-3 flex items-center gap-2 text-sm text-purple-600 hover:text-purple-700"
              >
                <Plus className="w-4 h-4" />
                Thêm triệu chứng
              </button>
            </div>

            {/* Duration */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Bạn có triệu chứng này trong bao lâu?
              </label>
              <select
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                value={duration}
                onChange={(e) => setDuration(e.target.value)}
              >
                <option value="">Chọn khoảng thời gian</option>
                <option value="few_hours">Vài giờ</option>
                <option value="1_day">1 ngày</option>
                <option value="2_3_days">2-3 ngày</option>
                <option value="1_week">1 tuần</option>
                <option value="more_week">Hơn 1 tuần</option>
              </select>
            </div>

            {/* Severity */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Mức độ nghiêm trọng
              </label>
              <div className="grid grid-cols-3 gap-3">
                {severityOptions.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setSeverity(option.value)}
                    className={`p-4 border-2 rounded-lg text-center transition-all ${
                      severity === option.value
                        ? 'border-purple-600 bg-purple-50'
                        : 'border-gray-200 hover:border-purple-300'
                    }`}
                  >
                    <div className={`font-semibold ${option.color}`}>
                      {option.label}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Submit */}
            <Button
              type="submit"
              disabled={loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Đang phân tích...
                </>
              ) : (
                <>
                  <Send className="w-5 h-5 mr-2" />
                  Kiểm tra triệu chứng
                </>
              )}
            </Button>
          </form>
        ) : (
          /* Results */
          <div className="space-y-6">
            {/* Risk Level */}
            <div className={`p-6 border-2 rounded-lg ${getRiskColor(result?.risk_level)}`}>
              <div className="flex items-center gap-3 mb-4">
                {getRiskIcon(result?.risk_level)}
                <div>
                  <h3 className="font-bold text-lg">
                    Mức độ nguy cơ: {result?.risk_level === 'low' ? 'Thấp' : result?.risk_level === 'medium' ? 'Trung bình' : 'Cao'}
                  </h3>
                  <p className="text-sm opacity-90">
                    {result?.risk_description || 'Đánh giá dựa trên các triệu chứng bạn mô tả'}
                  </p>
                </div>
              </div>
            </div>

            {/* Possible Conditions */}
            {result?.possible_conditions && result.possible_conditions.length > 0 && (
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Các bệnh có thể gặp:</h3>
                <div className="space-y-3">
                  {result.possible_conditions.map((condition, index) => (
                    <div key={index} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold text-gray-900">{condition.name}</h4>
                        {condition.probability && (
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                            {condition.probability}%
                          </span>
                        )}
                      </div>
                      {condition.description && (
                        <p className="text-sm text-gray-600 mb-2">{condition.description}</p>
                      )}
                      {condition.symptoms_matched && (
                        <div className="flex flex-wrap gap-1">
                          {condition.symptoms_matched.map((sym, i) => (
                            <Badge key={i} text={sym} />
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {result?.recommendations && result.recommendations.length > 0 && (
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Khuyến nghị:</h3>
                <ul className="space-y-2">
                  {result.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start gap-3 p-3 bg-blue-50 rounded-lg">
                      <CheckCircle className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-700">{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* When to see doctor */}
            {result?.when_to_see_doctor && (
              <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-orange-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <h4 className="font-semibold text-orange-900 mb-2">Khi nào cần gặp bác sĩ:</h4>
                    <p className="text-sm text-orange-800">{result.when_to_see_doctor}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 pt-4 border-t">
              <Button
                onClick={handleReset}
                variant="outline"
                className="flex-1"
              >
                Kiểm tra lại
              </Button>
              <Button
                onClick={() => window.location.href = '/patient/appointments/book'}
                className="flex-1"
              >
                Đặt lịch khám
              </Button>
            </div>

            {/* Disclaimer */}
            <div className="text-xs text-gray-500 text-center pt-4 border-t">
              Thông tin trên chỉ mang tính chất tham khảo. Vui lòng tham khảo ý kiến bác sĩ để có chẩn đoán chính xác.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}