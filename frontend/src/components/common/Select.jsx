import React, { forwardRef } from 'react';
import { ChevronDown, AlertCircle } from 'lucide-react';

const Select = forwardRef(({
  label,
  error,
  hint,
  options = [],
  placeholder = 'Chọn...',
  required = false,
  disabled = false,
  className = '',
  containerClassName = '',
  ...props
}, ref) => {
  const hasError = !!error;
  
  const selectClasses = `
    w-full px-4 py-2 text-base border rounded-lg appearance-none
    focus:outline-none focus:ring-2 focus:ring-offset-1 transition-all duration-200
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
        <select
          ref={ref}
          disabled={disabled}
          className={selectClasses}
          {...props}
        >
          {placeholder && <option value="">{placeholder}</option>}
          {options.map((option, index) => (
            <option key={index} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        
        <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none">
          {hasError ? (
            <AlertCircle size={18} className="text-red-500" />
          ) : (
            <ChevronDown size={18} className="text-gray-400" />
          )}
        </div>
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

Select.displayName = 'Select';

export default Select;