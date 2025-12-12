import React, { useEffect } from 'react';
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react';

const Toast = ({ 
  type = 'info',
  title,
  message,
  duration = 4000,
  onClose,
  position = 'top-right',
}) => {
  useEffect(() => {
    if (duration && onClose) {
      const timer = setTimeout(onClose, duration);
      return () => clearTimeout(timer);
    }
  }, [duration, onClose]);
  
  const types = {
    success: {
      icon: CheckCircle,
      bgColor: 'bg-green-50',
      borderColor: 'border-green-500',
      iconColor: 'text-green-500',
      textColor: 'text-green-800',
    },
    error: {
      icon: AlertCircle,
      bgColor: 'bg-red-50',
      borderColor: 'border-red-500',
      iconColor: 'text-red-500',
      textColor: 'text-red-800',
    },
    warning: {
      icon: AlertTriangle,
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-500',
      iconColor: 'text-yellow-500',
      textColor: 'text-yellow-800',
    },
    info: {
      icon: Info,
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-500',
      iconColor: 'text-blue-500',
      textColor: 'text-blue-800',
    },
  };
  
  const positions = {
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'top-center': 'top-4 left-1/2 -translate-x-1/2',
    'bottom-center': 'bottom-4 left-1/2 -translate-x-1/2',
  };
  
  const config = types[type];
  const Icon = config.icon;
  
  return (
    <div 
      className={`
        fixed ${positions[position]} z-50 
        flex items-start gap-3 p-4 min-w-[320px] max-w-md
        ${config.bgColor} border-l-4 ${config.borderColor}
        rounded-lg shadow-lg animate-slide-in
      `}
    >
      <Icon className={`flex-shrink-0 ${config.iconColor}`} size={20} />
      
      <div className="flex-1">
        {title && (
          <h4 className={`font-semibold ${config.textColor} mb-1`}>
            {title}
          </h4>
        )}
        {message && (
          <p className={`text-sm ${config.textColor}`}>
            {message}
          </p>
        )}
      </div>
      
      {onClose && (
        <button
          onClick={onClose}
          className={`flex-shrink-0 ${config.iconColor} hover:opacity-75 transition-opacity`}
        >
          <X size={18} />
        </button>
      )}
    </div>
  );
};

export default Toast;