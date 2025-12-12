import React, { useState, useEffect } from 'react';
import { Search, X } from 'lucide-react';
import { useDebounce } from '../../hooks/useDebounce';

const SearchBar = ({
  placeholder = 'Tìm kiếm...',
  value: controlledValue,
  onChange,
  onSearch,
  debounce = 500,
  showClearButton = true,
  className = '',
  ...props
}) => {
  const [internalValue, setInternalValue] = useState(controlledValue || '');
  const debouncedValue = useDebounce(internalValue, debounce);
  
  useEffect(() => {
    if (controlledValue !== undefined) {
      setInternalValue(controlledValue);
    }
  }, [controlledValue]);
  
  useEffect(() => {
    onSearch?.(debouncedValue);
  }, [debouncedValue, onSearch]);
  
  const handleChange = (e) => {
    const newValue = e.target.value;
    setInternalValue(newValue);
    onChange?.(newValue);
  };
  
  const handleClear = () => {
    setInternalValue('');
    onChange?.('');
    onSearch?.('');
  };
  
  return (
    <div className={`relative ${className}`}>
      <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
        <Search size={18} />
      </div>
      
      <input
        type="text"
        value={internalValue}
        onChange={handleChange}
        placeholder={placeholder}
        className="w-full pl-10 pr-10 py-2 text-base border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
        {...props}
      />
      
      {showClearButton && internalValue && (
        <button
          onClick={handleClear}
          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <X size={18} />
        </button>
      )}
    </div>
  );
};

export default SearchBar;