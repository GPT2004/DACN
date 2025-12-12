import React from 'react';
import { User } from 'lucide-react';

const Avatar = ({ 
  src, 
  alt = 'Avatar', 
  name,
  size = 'md',
  shape = 'circle',
  status,
  className = '',
  ...props 
}) => {
  const sizes = {
    xs: 'w-6 h-6 text-xs',
    sm: 'w-8 h-8 text-sm',
    md: 'w-10 h-10 text-base',
    lg: 'w-12 h-12 text-lg',
    xl: 'w-16 h-16 text-xl',
    '2xl': 'w-20 h-20 text-2xl',
  };
  
  const shapes = {
    circle: 'rounded-full',
    square: 'rounded-lg',
  };
  
  const getInitials = (name) => {
    if (!name) return '';
    const words = name.trim().split(' ');
    if (words.length === 1) return words[0].charAt(0).toUpperCase();
    return (words[0].charAt(0) + words[words.length - 1].charAt(0)).toUpperCase();
  };
  
  const avatarClasses = `
    relative inline-flex items-center justify-center
    ${sizes[size]}
    ${shapes[shape]}
    ${!src ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white font-semibold' : ''}
    ${className}
  `.trim();
  
  const statusSizes = {
    xs: 'w-1.5 h-1.5',
    sm: 'w-2 h-2',
    md: 'w-2.5 h-2.5',
    lg: 'w-3 h-3',
    xl: 'w-3.5 h-3.5',
    '2xl': 'w-4 h-4',
  };
  
  const statusColors = {
    online: 'bg-green-500',
    offline: 'bg-gray-400',
    busy: 'bg-red-500',
    away: 'bg-yellow-500',
  };
  
  return (
    <div className={avatarClasses} {...props}>
      {src ? (
        <img 
          src={src} 
          alt={alt} 
          className={`w-full h-full object-cover ${shapes[shape]}`}
        />
      ) : name ? (
        <span>{getInitials(name)}</span>
      ) : (
        <User size={size === 'xs' ? 12 : size === 'sm' ? 16 : size === 'lg' ? 24 : size === 'xl' ? 32 : size === '2xl' ? 40 : 20} />
      )}
      
      {status && (
        <span 
          className={`
            absolute bottom-0 right-0 block ${statusSizes[size]} 
            ${statusColors[status]} ${shapes[shape]} border-2 border-white
          `}
        />
      )}
    </div>
  );
};

export default Avatar;