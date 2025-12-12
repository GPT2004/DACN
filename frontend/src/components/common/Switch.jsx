import React from 'react';

const Switch = ({
  checked = false,
  onChange,
  disabled = false,
  label,
  size = 'md', // 'sm' | 'md' | 'lg'
  className = '',
}) => {
  const sizes = {
    sm: { track: 'w-8 h-4', thumb: 'w-3 h-3', translate: 'translate-x-4' },
    md: { track: 'w-11 h-6', thumb: 'w-5 h-5', translate: 'translate-x-5' },
    lg: { track: 'w-14 h-7', thumb: 'w-6 h-6', translate: 'translate-x-7' },
  };

  const sizeClasses = sizes[size];

  return (
    <label className={`inline-flex items-center gap-3 cursor-pointer ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}>
      <div className="relative">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange?.(e.target.checked)}
          disabled={disabled}
          className="sr-only peer"
        />
        <div className={`
          ${sizeClasses.track} rounded-full transition-colors
          ${checked ? 'bg-blue-600' : 'bg-gray-300'}
          peer-focus:ring-2 peer-focus:ring-blue-500 peer-focus:ring-offset-1
        `}>
          <div className={`
            ${sizeClasses.thumb} bg-white rounded-full shadow-md
            absolute top-0.5 left-0.5 transition-transform
            ${checked ? sizeClasses.translate : 'translate-x-0'}
          `} />
        </div>
      </div>
      {label && (
        <span className="text-sm text-gray-700 select-none">
          {label}
        </span>
      )}
    </label>
  );
};

export default Switch;