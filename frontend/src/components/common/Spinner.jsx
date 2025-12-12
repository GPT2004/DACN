import React from 'react';

const Spinner = ({
  size = 'md',
  color = 'blue',
  className = '',
}) => {
  const sizes = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-2',
    lg: 'w-12 h-12 border-3',
    xl: 'w-16 h-16 border-4',
  };

  const colors = {
    blue: 'border-blue-600 border-t-transparent',
    green: 'border-green-600 border-t-transparent',
    red: 'border-red-600 border-t-transparent',
    yellow: 'border-yellow-600 border-t-transparent',
    gray: 'border-gray-600 border-t-transparent',
  };

  return (
    <div
      className={`
        ${sizes[size]} ${colors[color]}
        rounded-full animate-spin ${className}
      `}
    />
  );
};

export default Spinner;