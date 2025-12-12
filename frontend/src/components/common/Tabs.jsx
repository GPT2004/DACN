import React, { useState } from 'react';

const Tabs = ({ 
  items = [], 
  defaultActiveKey, 
  activeKey: controlledActiveKey,
  onChange,
  className = '',
  type = 'line' // 'line' | 'card'
}) => {
  const [internalActiveKey, setInternalActiveKey] = useState(
    defaultActiveKey || items[0]?.key
  );

  const activeKey = controlledActiveKey !== undefined ? controlledActiveKey : internalActiveKey;

  const handleTabClick = (key) => {
    if (controlledActiveKey === undefined) {
      setInternalActiveKey(key);
    }
    onChange?.(key);
  };

  const activeItem = items.find(item => item.key === activeKey);

  const tabClasses = {
    line: `
      border-b-2 transition-colors
      ${activeKey === activeItem?.key
        ? 'border-blue-600 text-blue-600 font-medium'
        : 'border-transparent text-gray-600 hover:text-blue-600 hover:border-gray-300'
      }
    `,
    card: `
      border rounded-t-lg transition-colors
      ${activeKey === activeItem?.key
        ? 'bg-white border-gray-300 border-b-white text-blue-600 font-medium -mb-px'
        : 'bg-gray-50 border-transparent text-gray-600 hover:text-blue-600'
      }
    `
  };

  return (
    <div className={className}>
      {/* Tab headers */}
      <div className={`flex gap-1 ${type === 'line' ? 'border-b border-gray-200' : ''}`}>
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.key}
              onClick={() => handleTabClick(item.key)}
              disabled={item.disabled}
              className={`
                flex items-center gap-2 px-4 py-2.5 text-sm
                disabled:opacity-50 disabled:cursor-not-allowed
                ${tabClasses[type]}
              `}
            >
              {Icon && <Icon size={18} />}
              {item.label}
              {item.badge && (
                <span className="ml-1 px-2 py-0.5 bg-blue-100 text-blue-600 rounded-full text-xs font-medium">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div className="py-4">
        {activeItem?.content}
      </div>
    </div>
  );
};

export default Tabs;