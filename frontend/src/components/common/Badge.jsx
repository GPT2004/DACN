import React from 'react';

const Badge = ({ 
  text, 
  variant = 'default',
  size = 'md',
  dot = false,
  className = '',
  ...props 
}) => {
  const variants = {
    default: 'bg-gray-100 text-gray-800',
    primary: 'bg-blue-100 text-blue-800',
    success: 'bg-green-100 text-green-800',
    warning: 'bg-yellow-100 text-yellow-800',
    danger: 'bg-red-100 text-red-800',
    info: 'bg-cyan-100 text-cyan-800',
  };
  
  const sizes = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base',
  };
  
  const badgeClasses = `
    inline-flex items-center font-medium rounded-full
    ${variants[variant]}
    ${sizes[size]}
    ${className}
  `.trim();
  
  return (
    <span className={badgeClasses} {...props}>
      {dot && (
        <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
          variant === 'success' ? 'bg-green-600' :
          variant === 'warning' ? 'bg-yellow-600' :
          variant === 'danger' ? 'bg-red-600' :
          variant === 'info' ? 'bg-cyan-600' :
          variant === 'primary' ? 'bg-blue-600' :
          'bg-gray-600'
        }`} />
      )}
      {text}
    </span>
  );
};

export default Badge;