// frontend/src/components/common/Card.jsx
import React from 'react';

const Card = ({ 
  children, 
  title, 
  subtitle,
  actions,
  footer,
  className = '', 
  padding = 'default',
  hoverable = false,
  bordered = true,
  ...props 
}) => {
  const paddingStyles = {
    none: '',
    sm: 'p-4',
    default: 'p-6',
    lg: 'p-8',
  };
  
  const cardClasses = `
    bg-white rounded-lg shadow-sm
    ${bordered ? 'border border-gray-200' : ''}
    ${hoverable ? 'hover:shadow-md transition-shadow duration-200 cursor-pointer' : ''}
    ${className}
  `.trim();
  
  return (
    <div className={cardClasses} {...props}>
      {(title || subtitle || actions) && (
        <div className={`${paddingStyles[padding]} border-b border-gray-200`}>
          <div className="flex items-start justify-between">
            <div>
              {title && (
                <h3 className="text-lg font-semibold text-gray-900">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="mt-1 text-sm text-gray-500">
                  {subtitle}
                </p>
              )}
            </div>
            {actions && (
              <div className="flex items-center gap-2">
                {actions}
              </div>
            )}
          </div>
        </div>
      )}
      
      <div className={paddingStyles[padding]}>
        {children}
      </div>
      
      {footer && (
        <div className={`${paddingStyles[padding]} border-t border-gray-200 bg-gray-50 rounded-b-lg`}>
          {footer}
        </div>
      )}
    </div>
  );
};

export default Card;