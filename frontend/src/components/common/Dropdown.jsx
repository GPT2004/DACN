/* eslint-disable no-unused-vars */
import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';

const Dropdown = ({
  trigger,
  items = [],
  placement = 'bottom-left', // 'bottom-left' | 'bottom-right' | 'top-left' | 'top-right'
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const placements = {
    'bottom-left': 'top-full left-0 mt-2',
    'bottom-right': 'top-full right-0 mt-2',
    'top-left': 'bottom-full left-0 mb-2',
    'top-right': 'bottom-full right-0 mb-2',
  };

  const handleItemClick = (item) => {
    item.onClick?.();
    if (!item.keepOpen) {
      setIsOpen(false);
    }
  };

  return (
    <div ref={dropdownRef} className={`relative ${className}`}>
      <div onClick={() => setIsOpen(!isOpen)}>
        {trigger}
      </div>

      {isOpen && (
        <div className={`
          absolute ${placements[placement]} z-50
          min-w-[200px] bg-white rounded-lg shadow-lg border border-gray-200
          py-1 animate-fade-in
        `}>
          {items.map((item, index) => {
            if (item.divider) {
              return <div key={index} className="border-t border-gray-200 my-1" />;
            }

            const Icon = item.icon;

            return (
              <button
                key={index}
                onClick={() => handleItemClick(item)}
                disabled={item.disabled}
                className={`
                  w-full flex items-center gap-3 px-4 py-2 text-sm text-left
                  transition-colors
                  ${item.disabled 
                    ? 'opacity-50 cursor-not-allowed' 
                    : item.danger
                      ? 'text-red-600 hover:bg-red-50'
                      : 'text-gray-700 hover:bg-gray-100'
                  }
                `}
              >
                {Icon && <Icon size={16} />}
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default Dropdown;
