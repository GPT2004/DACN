import React, { forwardRef } from 'react';
import { Calendar, AlertCircle } from 'lucide-react';

const DatePicker = forwardRef(({
  label,
  error,
  hint,
  required = false,
  disabled = false,
  className = '',
  containerClassName = '',
  min,
  max,
  ...props
}, ref) => {
  const hasError = !!error;
  
  const inputClasses = `
    w-full pl-10 pr-4 py-2 text-base border rounded-lg transition-all duration-200
    focus:outline-none focus:ring-2 focus:ring-offset-1
    disabled:bg-gray-100 disabled:cursor-not-allowed disabled:text-gray-500
    ${hasError 
      ? 'border-red-500 focus:border-red-500 focus:ring-red-500' 
      : 'border-gray-300 focus:border-blue-500 focus:ring-blue-500'
    }
    ${className}
  `.trim();
  
  return (
    <div className={`w-full ${containerClassName}`}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      
      <div className="relative">
        <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
          <Calendar size={18} />
        </div>
        
        <input
          ref={ref}
          type="date"
          disabled={disabled}
          min={min}
          max={max}
          className={inputClasses}
          {...props}
        />
        
        {hasError && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-red-500">
            <AlertCircle size={18} />
          </div>
        )}
      </div>
      
      {hint && !hasError && (
        <p className="mt-1 text-sm text-gray-500">{hint}</p>
      )}
      
      {hasError && (
        <p className="mt-1 text-sm text-red-500 flex items-center gap-1">
          <AlertCircle size={14} />
          {error}
        </p>
      )}
    </div>
  );
});

DatePicker.displayName = 'DatePicker';

export default DatePicker;