import React from 'react';
import { FileQuestion, Inbox, Search } from 'lucide-react';
import Button from './Button';

const EmptyState = ({
  icon: Icon,
  title = 'Không có dữ liệu',
  description,
  action,
  actionLabel,
  onAction,
  type = 'default',
  className = '',
}) => {
  const defaultIcons = {
    default: Inbox,
    search: Search,
    error: FileQuestion,
  };
  
  const DisplayIcon = Icon || defaultIcons[type];
  
  return (
    <div className={`flex flex-col items-center justify-center py-12 px-4 ${className}`}>
      <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
        <DisplayIcon size={32} className="text-gray-400" />
      </div>
      
      <h3 className="text-lg font-semibold text-gray-900 mb-2">
        {title}
      </h3>
      
      {description && (
        <p className="text-sm text-gray-500 text-center max-w-md mb-6">
          {description}
        </p>
      )}
      
      {(action || (actionLabel && onAction)) && (
        <div>
          {action || (
            <Button variant="primary" onClick={onAction}>
              {actionLabel}
            </Button>
          )}
        </div>
      )}
    </div>
  );
};

export default EmptyState;