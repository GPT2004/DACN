import React from 'react';

const Divider = ({ 
  orientation = 'horizontal',
  children,
  className = '',
  dashed = false 
}) => {
  const baseClasses = dashed ? 'border-dashed' : '';

  if (orientation === 'vertical') {
    return (
      <div className={`border-l border-gray-200 ${baseClasses} ${className}`} />
    );
  }

  if (children) {
    return (
      <div className={`flex items-center gap-4 my-4 ${className}`}>
        <div className={`flex-1 border-t border-gray-200 ${baseClasses}`} />
        <span className="text-sm text-gray-500">{children}</span>
        <div className={`flex-1 border-t border-gray-200 ${baseClasses}`} />
      </div>
    );
  }

  return (
    <div className={`border-t border-gray-200 my-4 ${baseClasses} ${className}`} />
  );
};

export default Divider;