import React from 'react';

const Progress = ({
  percent = 0,
  showInfo = true,
  status, // 'success' | 'error' | 'warning'
  size = 'md', // 'sm' | 'md' | 'lg'
  strokeColor,
  className = '',
}) => {
  const sizes = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  const getColor = () => {
    if (strokeColor) return strokeColor;
    if (status === 'success') return 'bg-green-500';
    if (status === 'error') return 'bg-red-500';
    if (status === 'warning') return 'bg-yellow-500';
    return 'bg-blue-600';
  };

  return (
    <div className={className}>
      <div className="flex items-center gap-3">
        <div className={`flex-1 bg-gray-200 rounded-full overflow-hidden ${sizes[size]}`}>
          <div
            className={`h-full ${getColor()} transition-all duration-300 ease-out`}
            style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
          />
        </div>
        {showInfo && (
          <span className="text-sm text-gray-600 font-medium min-w-[3rem] text-right">
            {Math.round(percent)}%
          </span>
        )}
      </div>
    </div>
  );
};

export default Progress;