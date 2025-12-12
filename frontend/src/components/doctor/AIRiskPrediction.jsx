// src/components/doctor/AIRiskPrediction.jsx
import React, { useState } from 'react';
import { Activity, CheckCircle } from 'lucide-react';

const AIRiskPrediction = ({ patient }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeMedicalRecord = () => {
    setLoading(true);
    setTimeout(() => {
      setPrediction({
        riskLevel: 'medium',
        riskScore: 45,
        factors: [
          { name: 'Tuổi cao', impact: 'high' },
          { name: 'Huyết áp cao', impact: 'medium' },
          { name: 'Tiền sử gia đình', impact: 'medium' }
        ],
        recommendations: [
          'Theo dõi huyết áp thường xuyên',
          'Chế độ ăn ít muối',
          'Tập thể dục nhẹ nhàng'
        ]
      });
      setLoading(false);
    }, 2000);
  };

  const getRiskColor = (level) => {
    const colors = {
      low: 'text-green-600 bg-green-100',
      medium: 'text-yellow-600 bg-yellow-100',
      high: 'text-red-600 bg-red-100'
    };
    return colors[level] || colors.medium;
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900">Dự đoán Nguy cơ (AI)</h3>
        <button
          onClick={analyzeMedicalRecord}
          disabled={loading}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400"
        >
          {loading ? 'Đang phân tích...' : 'Phân tích'}
        </button>
      </div>
      {prediction && (
        <div className="space-y-4">
          <div className={`p-4 rounded-lg ${getRiskColor(prediction.riskLevel)}`}>
            <div className="flex items-center justify-between">
              <span className="font-medium">Mức độ nguy cơ</span>
              <span className="text-2xl font-bold">{prediction.riskScore}%</span>
            </div>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Yếu tố nguy cơ</h4>
            <div className="space-y-2">
              {prediction.factors.map((factor, idx) => (
                <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <span className="text-sm text-gray-700">{factor.name}</span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    factor.impact === 'high' ? 'bg-red-100 text-red-700' :
                    factor.impact === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-green-100 text-green-700'
                  }`}>
                    {factor.impact}
                  </span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Khuyến nghị</h4>
            <ul className="space-y-1">
              {prediction.recommendations.map((rec, idx) => (
                <li key={idx} className="text-sm text-gray-700 flex items-start">
                  <CheckCircle className="w-4 h-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
      {!prediction && !loading && (
        <div className="text-center py-8 text-gray-500">
          <Activity className="w-12 h-12 mx-auto mb-2 text-gray-300" />
          <p>Nhấn "Phân tích" để AI dự đoán nguy cơ</p>
        </div>
      )}
    </div>
  );
};

export default AIRiskPrediction;