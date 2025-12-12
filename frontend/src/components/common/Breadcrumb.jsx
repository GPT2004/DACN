import React from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight, Home } from 'lucide-react';

const Breadcrumb = ({ items = [], separator, className = '' }) => {
  if (items.length === 0) return null;

  const Separator = separator || (() => <ChevronRight size={16} className="text-gray-400" />);

  return (
    <nav className={`flex items-center gap-2 text-sm ${className}`}>
      <Link
        to="/"
        className="flex items-center gap-1 text-gray-600 hover:text-blue-600 transition-colors"
      >
        <Home size={16} />
      </Link>

      {items.map((item, index) => {
        const isLast = index === items.length - 1;

        return (
          <React.Fragment key={index}>
            <Separator />
            {isLast ? (
              <span className="text-gray-900 font-medium">{item.label}</span>
            ) : (
              <Link
                to={item.path}
                className="text-gray-600 hover:text-blue-600 transition-colors"
              >
                {item.label}
              </Link>
            )}
          </React.Fragment>
        );
      })}
    </nav>
  );
};

export default Breadcrumb;