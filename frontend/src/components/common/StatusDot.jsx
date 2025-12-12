import React from 'react';

const StatusDot = ({
  status = 'default',
  text,
  size = 'md',
  pulse = false,
  className = '',
}) => {
  const statuses = {
    success: 'bg-green-500',
    error: 'bg-red-500',
    warning: 'bg-yellow-500',
    info: 'bg-blue-500',
    default: 'bg-gray-500',
  };

  const sizes = {
    sm: 'w-2 h-2',
    md: 'w-2.5 h-2.5',
    lg: 'w-3 h-3',
  };

  return (
    <div className={`inline-flex items-center gap-2 ${className}`}>
      <div className="relative">
        <span
          className={`
            block rounded-full ${statuses[status]} ${sizes[size]}
          `}
        />
        {pulse && (
          <span
            className={`
              absolute inset-0 rounded-full ${statuses[status]} 
              animate-ping opacity-75
            `}
          />
        )}
      </div>
      {text && (
        <span className="text-sm text-gray-700">{text}</span>
      )}
    </div>
  );
};

export default StatusDot;