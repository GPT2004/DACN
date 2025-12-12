import React, { forwardRef } from 'react';
import { AlertCircle } from 'lucide-react';

const Textarea = forwardRef(({
  label,
  error,
  hint,
  required = false,
  disabled = false,
  className = '',
  containerClassName = '',
  rows = 4,
  maxLength,
  showCount = false,
  ...props
}, ref) => {
  const hasError = !!error;
  const [value, setValue] = React.useState(props.value || '');

  const textareaClasses = `
    w-full px-4 py-2 text-base border rounded-lg resize-y
    focus:outline-none focus:ring-2 focus:ring-offset-1 transition-all duration-200
    disabled:bg-gray-100 disabled:cursor-not-allowed disabled:text-gray-500
    ${hasError 
      ? 'border-red-500 focus:border-red-500 focus:ring-red-500' 
      : 'border-gray-300 focus:border-blue-500 focus:ring-blue-500'
    }
    ${className}
  `.trim();

  const handleChange = (e) => {
    setValue(e.target.value);
    props.onChange?.(e);
  };

  return (
    <div className={`w-full ${containerClassName}`}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      
      <textarea
        ref={ref}
        rows={rows}
        disabled={disabled}
        maxLength={maxLength}
        className={textareaClasses}
        value={value}
        onChange={handleChange}
        {...props}
      />
      
      <div className="flex items-center justify-between mt-1">
        <div className="flex-1">
          {hint && !hasError && (
            <p className="text-sm text-gray-500">{hint}</p>
          )}
          
          {hasError && (
            <p className="text-sm text-red-500 flex items-center gap-1">
              <AlertCircle size={14} />
              {error}
            </p>
          )}
        </div>
        
        {showCount && maxLength && (
          <span className="text-sm text-gray-500">
            {value.length}/{maxLength}
          </span>
        )}
      </div>
    </div>
  );
});

Textarea.displayName = 'Textarea';

export default Textarea;