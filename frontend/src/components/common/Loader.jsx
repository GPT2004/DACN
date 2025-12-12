import React from 'react';
import { Loader2 } from 'lucide-react';

const Loader = ({ 
  size = 'md', 
  text,
  fullScreen = false,
  overlay = false,
  className = '' 
}) => {
  const sizes = {
    sm: 16,
    md: 24,
    lg: 32,
    xl: 48,
  };
  
  const loader = (
    <div className={`flex flex-col items-center justify-center gap-3 ${className}`}>
      <Loader2 className="animate-spin text-blue-600" size={sizes[size]} />
      {text && (
        <p className="text-sm text-gray-600 font-medium">{text}</p>
      )}
    </div>
  );
  
  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white">
        {loader}
      </div>
    );
  }
  
  if (overlay) {
    return (
      <div className="absolute inset-0 z-10 flex items-center justify-center bg-white bg-opacity-75">
        {loader}
      </div>
    );
  }
  
  return loader;
};

export default Loader;
