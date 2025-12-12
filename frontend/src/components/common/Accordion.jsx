import React, { useState } from 'react';
import { ChevronDown } from 'lucide-react';

const Accordion = ({
  items = [],
  defaultActiveKey,
  onChange,
  className = '',
}) => {
  const [activeKeys, setActiveKeys] = useState(
    defaultActiveKey ? [defaultActiveKey] : []
  );

  const toggleItem = (key) => {
    const newActiveKeys = activeKeys.includes(key)
      ? activeKeys.filter(k => k !== key)
      : [...activeKeys, key];
    
    setActiveKeys(newActiveKeys);
    onChange?.(newActiveKeys);
  };

  return (
    <div className={`border border-gray-200 rounded-lg overflow-hidden ${className}`}>
      {items.map((item, index) => {
        const isActive = activeKeys.includes(item.key);
        
        return (
          <div
            key={item.key}
            className={index !== 0 ? 'border-t border-gray-200' : ''}
          >
            <button
              onClick={() => toggleItem(item.key)}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
            >
              <span className="font-medium text-gray-900">
                {item.label}
              </span>
              <ChevronDown
                size={20}
                className={`text-gray-500 transition-transform ${
                  isActive ? 'rotate-180' : ''
                }`}
              />
            </button>

            {isActive && (
              <div className="px-4 pb-4 text-gray-700 animate-fade-in">
                {item.content}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default Accordion;
