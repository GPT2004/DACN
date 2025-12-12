import React from 'react';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';
import Button from './Button';

const Pagination = ({
  currentPage = 1,
  totalPages = 1,
  pageSize = 10,
  totalItems = 0,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [10, 20, 50, 100],
  showTotal = true,
  showPageSize = true,
  className = '',
}) => {
  const handlePageChange = (page) => {
    if (page >= 1 && page <= totalPages && page !== currentPage) {
      onPageChange?.(page);
    }
  };
  
  const getPageNumbers = () => {
    const pages = [];
    const maxVisible = 7;
    
    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      if (currentPage <= 4) {
        for (let i = 1; i <= 5; i++) pages.push(i);
        pages.push('...');
        pages.push(totalPages);
      } else if (currentPage >= totalPages - 3) {
        pages.push(1);
        pages.push('...');
        for (let i = totalPages - 4; i <= totalPages; i++) pages.push(i);
      } else {
        pages.push(1);
        pages.push('...');
        for (let i = currentPage - 1; i <= currentPage + 1; i++) pages.push(i);
        pages.push('...');
        pages.push(totalPages);
      }
    }
    
    return pages;
  };
  
  const startItem = (currentPage - 1) * pageSize + 1;
  const endItem = Math.min(currentPage * pageSize, totalItems);
  
  return (
    <div className={`flex items-center justify-between gap-4 ${className}`}>
      {/* Total info */}
      {showTotal && (
        <div className="text-sm text-gray-600">
          Hiển thị <span className="font-medium">{startItem}</span> - <span className="font-medium">{endItem}</span> trong <span className="font-medium">{totalItems}</span> kết quả
        </div>
      )}
      
      <div className="flex items-center gap-4">
        {/* Page size selector */}
        {showPageSize && onPageSizeChange && (
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Hiển thị:</label>
            <select
              value={pageSize}
              onChange={(e) => onPageSizeChange(Number(e.target.value))}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {pageSizeOptions.map(size => (
                <option key={size} value={size}>{size}</option>
              ))}
            </select>
          </div>
        )}
        
        {/* Pagination controls */}
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            icon={ChevronsLeft}
            onClick={() => handlePageChange(1)}
            disabled={currentPage === 1}
          />
          <Button
            variant="ghost"
            size="sm"
            icon={ChevronLeft}
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
          />
          
          {getPageNumbers().map((page, index) => (
            page === '...' ? (
              <span key={`ellipsis-${index}`} className="px-3 py-1.5 text-gray-400">
                ...
              </span>
            ) : (
              <button
                key={page}
                onClick={() => handlePageChange(page)}
                className={`
                  px-3 py-1.5 text-sm font-medium rounded-lg transition-colors
                  ${page === currentPage
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                  }
                `}
              >
                {page}
              </button>
            )
          ))}
          
          <Button
            variant="ghost"
            size="sm"
            icon={ChevronRight}
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
          />
          <Button
            variant="ghost"
            size="sm"
            icon={ChevronsRight}
            onClick={() => handlePageChange(totalPages)}
            disabled={currentPage === totalPages}
          />
        </div>
      </div>
    </div>
  );
};

export default Pagination;