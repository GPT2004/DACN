import React from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import EmptyState from './EmptyState';
import Loader from './Loader';

const Table = ({
  columns = [],
  data = [],
  loading = false,
  emptyText = 'Không có dữ liệu',
  onSort,
  sortColumn,
  sortDirection,
  onRowClick,
  className = '',
  rowClassName,
}) => {
  const handleSort = (column) => {
    if (!column.sortable || !onSort) return;
    
    const newDirection = 
      sortColumn === column.key && sortDirection === 'asc' 
        ? 'desc' 
        : 'asc';
    
    onSort(column.key, newDirection);
  };
  
  if (loading) {
    return (
      <div className="border border-gray-200 rounded-lg p-8">
        <Loader text="Đang tải dữ liệu..." />
      </div>
    );
  }
  
  if (data.length === 0) {
    return (
      <div className="border border-gray-200 rounded-lg">
        <EmptyState title={emptyText} />
      </div>
    );
  }
  
  return (
    <div className={`overflow-x-auto border border-gray-200 rounded-lg ${className}`}>
      <table className="w-full">
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                className={`
                  px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider
                  ${column.sortable ? 'cursor-pointer hover:bg-gray-100 select-none' : ''}
                  ${column.align === 'center' ? 'text-center' : ''}
                  ${column.align === 'right' ? 'text-right' : ''}
                  ${column.className || ''}
                `}
                style={{ width: column.width }}
                onClick={() => handleSort(column)}
              >
                <div className="flex items-center gap-2">
                  <span>{column.title}</span>
                  {column.sortable && (
                    <div className="flex flex-col">
                      <ChevronUp 
                        size={12} 
                        className={
                          sortColumn === column.key && sortDirection === 'asc'
                            ? 'text-blue-600'
                            : 'text-gray-400'
                        }
                      />
                      <ChevronDown 
                        size={12} 
                        className={
                          sortColumn === column.key && sortDirection === 'desc'
                            ? 'text-blue-600'
                            : 'text-gray-400'
                        }
                      />
                    </div>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((row, rowIndex) => (
            <tr
              key={row.id || rowIndex}
              className={`
                hover:bg-gray-50 transition-colors
                ${onRowClick ? 'cursor-pointer' : ''}
                ${typeof rowClassName === 'function' ? rowClassName(row) : rowClassName || ''}
              `}
              onClick={() => onRowClick?.(row)}
            >
              {columns.map((column) => (
                <td
                  key={column.key}
                  className={`
                    px-6 py-4 text-sm text-gray-900
                    ${column.align === 'center' ? 'text-center' : ''}
                    ${column.align === 'right' ? 'text-right' : ''}
                    ${column.className || ''}
                  `}
                >
                  {column.render 
                    ? column.render(row[column.key], row, rowIndex)
                    : row[column.key]
                  }
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Table;