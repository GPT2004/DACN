import React from 'react';
import { CheckCircle, AlertCircle, AlertTriangle, Info, X } from 'lucide-react';

const Alert = ({
  type = 'info',
  title,
  message,
  closable = false,
  onClose,
  showIcon = true,
  className = '',
  children,
}) => {
  const types = {
    success: {
      icon: CheckCircle,
      bgColor: 'bg-green-50',
      borderColor: 'border-green-500',
      iconColor: 'text-green-500',
      titleColor: 'text-green-800',
      textColor: 'text-green-700',
    },
    error: {
      icon: AlertCircle,
      bgColor: 'bg-red-50',
      borderColor: 'border-red-500',
      iconColor: 'text-red-500',
      titleColor: 'text-red-800',
      textColor: 'text-red-700',
    },
    warning: {
      icon: AlertTriangle,
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-500',
      iconColor: 'text-yellow-500',
      titleColor: 'text-yellow-800',
      textColor: 'text-yellow-700',
    },
    info: {
      icon: Info,
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-500',
      iconColor: 'text-blue-500',
      titleColor: 'text-blue-800',
      textColor: 'text-blue-700',
    },
  };

  const config = types[type];
  const Icon = config.icon;

  return (
    <div
      className={`
        flex gap-3 p-4 rounded-lg border-l-4
        ${config.bgColor} ${config.borderColor}
        ${className}
      `}
    >
      {showIcon && (
        <Icon className={`flex-shrink-0 ${config.iconColor}`} size={20} />
      )}

      <div className="flex-1">
        {title && (
          <h4 className={`font-semibold ${config.titleColor} mb-1`}>
            {title}
          </h4>
        )}
        {message && (
          <p className={`text-sm ${config.textColor}`}>
            {message}
          </p>
        )}
        {children && (
          <div className={`text-sm ${config.textColor}`}>
            {children}
          </div>
        )}
      </div>

      {closable && (
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

export default Alert;