import React, { forwardRef } from 'react';

const Radio = forwardRef(({
  options = [],
  value,
  onChange,
  name,
  label,
  error,
  disabled = false,
  orientation = 'vertical', // 'vertical' | 'horizontal'
  className = '',
}, ref) => {
  return (
    <div className="flex flex-col gap-2">
      {label && (
        <label className="text-sm font-medium text-gray-700">
          {label}
        </label>
      )}
      
      <div className={`
        flex gap-4
        ${orientation === 'vertical' ? 'flex-col' : 'flex-row flex-wrap'}
        ${className}
      `}>
        {options.map((option, index) => (
          <label
            key={option.value || index}
            className={`
              inline-flex items-center gap-2 cursor-pointer
              ${disabled || option.disabled ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            <input
              ref={index === 0 ? ref : undefined}
              type="radio"
              name={name}
              value={option.value}
              checked={value === option.value}
              onChange={(e) => onChange?.(e.target.value)}
              disabled={disabled || option.disabled}
              className="w-4 h-4 text-blue-600 border-gray-300 focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700 select-none">
              {option.label}
            </span>
          </label>
        ))}
      </div>
      
      {error && (
        <span className="text-sm text-red-500">{error}</span>
      )}
    </div>
  );
});

Radio.displayName = 'Radio';

export default Radio;