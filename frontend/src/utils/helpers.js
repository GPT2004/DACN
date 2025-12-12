import dayjs from 'dayjs';
import { DATE_FORMAT, TIME_FORMAT, DATETIME_FORMAT } from './constants';

/**
 * Format date
 */
export const formatDate = (date, format = DATE_FORMAT) => {
  if (!date) return '';
  return dayjs(date).format(format);
};

/**
 * Format time
 */
export const formatTime = (time, format = TIME_FORMAT) => {
  if (!time) return '';
  return dayjs(time, 'HH:mm:ss').format(format);
};

/**
 * Format datetime
 */
export const formatDateTime = (datetime, format = DATETIME_FORMAT) => {
  if (!datetime) return '';
  return dayjs(datetime).format(format);
};

/**
 * Calculate age from date of birth
 */
export const calculateAge = (dob) => {
  if (!dob) return null;
  const today = dayjs();
  const birthDate = dayjs(dob);
  return today.diff(birthDate, 'year');
};

/**
 * Format currency (VND)
 */
export const formatCurrency = (amount) => {
  if (!amount && amount !== 0) return '0 ₫';
  return new Intl.NumberFormat('vi-VN', {
    style: 'currency',
    currency: 'VND',
  }).format(amount);
};

/**
 * Format number
 */
export const formatNumber = (number) => {
  if (!number && number !== 0) return '0';
  return new Intl.NumberFormat('vi-VN').format(number);
};

/**
 * Get initials from full name
 */
export const getInitials = (name) => {
  if (!name) return '';
  const words = name.trim().split(' ');
  if (words.length === 1) return words[0].charAt(0).toUpperCase();
  return (words[0].charAt(0) + words[words.length - 1].charAt(0)).toUpperCase();
};

/**
 * Generate random color for avatar
 */
export const getRandomColor = (name) => {
  const colors = [
    '#f56a00', '#7265e6', '#ffbf00', '#00a2ae',
    '#1890ff', '#52c41a', '#faad14', '#eb2f96',
  ];
  const index = name ? name.charCodeAt(0) % colors.length : 0;
  return colors[index];
};

/**
 * Truncate text
 */
export const truncateText = (text, maxLength = 50) => {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

/**
 * Check if date is today
 */
export const isToday = (date) => {
  return dayjs(date).isSame(dayjs(), 'day');
};

/**
 * Check if date is past
 */
export const isPast = (date) => {
  return dayjs(date).isBefore(dayjs(), 'day');
};

/**
 * Check if date is future
 */
export const isFuture = (date) => {
  return dayjs(date).isAfter(dayjs(), 'day');
};

/**
 * Get date range for filters
 */
export const getDateRange = (range) => {
  const today = dayjs();
  
  switch (range) {
    case 'today':
      return {
        startDate: today.startOf('day').toISOString(),
        endDate: today.endOf('day').toISOString(),
      };
    case 'yesterday':
      return {
        startDate: today.subtract(1, 'day').startOf('day').toISOString(),
        endDate: today.subtract(1, 'day').endOf('day').toISOString(),
      };
    case 'thisWeek':
      return {
        startDate: today.startOf('week').toISOString(),
        endDate: today.endOf('week').toISOString(),
      };
    case 'lastWeek':
      return {
        startDate: today.subtract(1, 'week').startOf('week').toISOString(),
        endDate: today.subtract(1, 'week').endOf('week').toISOString(),
      };
    case 'thisMonth':
      return {
        startDate: today.startOf('month').toISOString(),
        endDate: today.endOf('month').toISOString(),
      };
    case 'lastMonth':
      return {
        startDate: today.subtract(1, 'month').startOf('month').toISOString(),
        endDate: today.subtract(1, 'month').endOf('month').toISOString(),
      };
    case 'last7Days':
      return {
        startDate: today.subtract(7, 'day').startOf('day').toISOString(),
        endDate: today.endOf('day').toISOString(),
      };
    case 'last30Days':
      return {
        startDate: today.subtract(30, 'day').startOf('day').toISOString(),
        endDate: today.endOf('day').toISOString(),
      };
    default:
      return null;
  }
};

/**
 * Debounce function
 */
export const debounce = (func, wait = 300) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

/**
 * Deep clone object
 */
export const deepClone = (obj) => {
  return JSON.parse(JSON.stringify(obj));
};

/**
 * Check if object is empty
 */
export const isEmpty = (obj) => {
  if (!obj) return true;
  if (Array.isArray(obj)) return obj.length === 0;
  if (typeof obj === 'object') return Object.keys(obj).length === 0;
  return !obj;
};

/**
 * Remove empty fields from object
 */
export const removeEmptyFields = (obj) => {
  return Object.keys(obj).reduce((acc, key) => {
    if (obj[key] !== null && obj[key] !== undefined && obj[key] !== '') {
      acc[key] = obj[key];
    }
    return acc;
  }, {});
};

/**
 * Sort array by field
 */
export const sortBy = (array, field, order = 'asc') => {
  return [...array].sort((a, b) => {
    const aVal = a[field];
    const bVal = b[field];
    
    if (order === 'asc') {
      return aVal > bVal ? 1 : -1;
    }
    return aVal < bVal ? 1 : -1;
  });
};

/**
 * Group array by field
 */
export const groupBy = (array, field) => {
  return array.reduce((acc, item) => {
    const key = item[field];
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(item);
    return acc;
  }, {});
};

/**
 * Get unique values from array
 */
export const getUniqueValues = (array, field) => {
  return [...new Set(array.map(item => item[field]))];
};

/**
 * Calculate percentage
 */
export const calculatePercentage = (value, total) => {
  if (!total || total === 0) return 0;
  return Math.round((value / total) * 100);
};

/**
 * Generate time slots
 */
export const generateTimeSlots = (startTime, endTime, duration = 20) => {
  const slots = [];
  let current = dayjs(`2000-01-01 ${startTime}`);
  const end = dayjs(`2000-01-01 ${endTime}`);
  
  while (current.isBefore(end)) {
    slots.push({
      start: current.format('HH:mm'),
      end: current.add(duration, 'minute').format('HH:mm'),
    });
    current = current.add(duration, 'minute');
  }
  
  return slots;
};

/**
 * Validate email
 */
export const validateEmail = (email) => {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
};

/**
 * Validate phone number (Vietnam)
 */
export const validatePhone = (phone) => {
  const re = /^(0|\+84)[0-9]{9,10}$/;
  return re.test(phone);
};

/**
 * Generate random ID
 */
export const generateId = (prefix = '') => {
  return `${prefix}${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Download file
 */
export const downloadFile = (url, filename) => {
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

/**
 * Copy to clipboard
 */
export const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    console.error('Failed to copy:', err);
    return false;
  }
};

/**
 * Format file size
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};