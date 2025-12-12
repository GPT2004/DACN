import React, { forwardRef } from 'react';
import { Check } from 'lucide-react';

const Checkbox = forwardRef(({
  label,
  checked,
  onChange,
  disabled = false,
  error,
  className = '',
  ...props
}, ref) => {
  return (
    <div className="flex flex-col gap-1">
      <label className={`inline-flex items-center gap-2 cursor-pointer ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
        <div className="relative">
          <input
            ref={ref}
            type="checkbox"
            checked={checked}
            onChange={onChange}
            disabled={disabled}
            className="sr-only peer"
            {...props}
          />
          <div className={`
            w-5 h-5 border-2 rounded transition-all
            peer-checked:bg-blue-600 peer-checked:border-blue-600
            peer-focus:ring-2 peer-focus:ring-blue-500 peer-focus:ring-offset-1
            ${error ? 'border-red-500' : 'border-gray-300'}
            ${className}
          `}>
            {checked && (
              <Check size={16} className="text-white absolute top-0 left-0" strokeWidth={3} />
            )}
          </div>
        </div>
        {label && (
          <span className="text-sm text-gray-700 select-none">
            {label}
          </span>
        )}
      </label>
      {error && (
        <span className="text-sm text-red-500">{error}</span>
      )}
    </div>
  );
});

Checkbox.displayName = 'Checkbox';

export default Checkbox;