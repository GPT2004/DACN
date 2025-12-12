// frontend/src/config/app.config.js
/**
 * Application Configuration
 * Các cấu hình chung của ứng dụng
 */

export const APP_CONFIG = {
  // App Info
  name: process.env.REACT_APP_NAME || 'Clinic Management System',
  version: process.env.REACT_APP_VERSION || '1.0.0',
  environment: process.env.REACT_APP_ENV || 'development',

  // Features flags
  features: {
    enableAI: process.env.REACT_APP_ENABLE_AI === 'true',
    enableNotifications: process.env.REACT_APP_ENABLE_NOTIFICATIONS === 'true',
    enableWebSocket: process.env.REACT_APP_ENABLE_WEBSOCKET === 'true',
  },

  // WebSocket
  webSocket: {
    url: process.env.REACT_APP_WS_URL || 'ws://localhost:3000',
    reconnectAttempts: 5,
    reconnectDelay: 3000, // 3 seconds
  },

  // Pagination
  pagination: {
    defaultPage: 1,
    defaultPageSize: 10,
    pageSizeOptions: [10, 20, 50, 100],
    showSizeChanger: true,
    showQuickJumper: true,
  },

  // Date & Time Formats
  dateFormat: 'DD/MM/YYYY',
  timeFormat: 'HH:mm',
  dateTimeFormat: 'DD/MM/YYYY HH:mm',
  
  // Locale
  locale: 'vi',
  timezone: 'Asia/Ho_Chi_Minh',

  // File Upload
  upload: {
    maxSize: parseInt(process.env.REACT_APP_MAX_FILE_SIZE) || 5242880, // 5MB
    allowedTypes: process.env.REACT_APP_ALLOWED_FILE_TYPES?.split(',') || [
      'image/jpeg',
      'image/png',
      'application/pdf',
    ],
    maxSizeDisplay: '5MB',
  },

  // Session
  session: {
    timeout: 30 * 60 * 1000, // 30 minutes
    warningTime: 5 * 60 * 1000, // 5 minutes before timeout
  },

  // Notifications
  notification: {
    duration: 4.5, // seconds
    placement: 'topRight',
    maxCount: 3,
  },

  // Table
  table: {
    defaultPageSize: 10,
    showSizeChanger: true,
    showQuickJumper: true,
    size: 'middle',
  },

  // Chart colors
  chartColors: [
    '#1890ff', // blue
    '#52c41a', // green
    '#faad14', // orange
    '#f5222d', // red
    '#722ed1', // purple
    '#13c2c2', // cyan
    '#eb2f96', // magenta
    '#fa8c16', // volcano
  ],

  // Map (if needed)
  map: {
    defaultCenter: {
      lat: 10.8231, // Ho Chi Minh City
      lng: 106.6297,
    },
    defaultZoom: 12,
  },

  // Debounce delays
  debounce: {
    search: 500,
    input: 300,
    resize: 200,
  },

  // Cache duration (ms)
  cache: {
    userProfile: 5 * 60 * 1000, // 5 minutes
    doctors: 10 * 60 * 1000, // 10 minutes
    medicines: 15 * 60 * 1000, // 15 minutes
  },

  // API retry configuration
  retry: {
    maxAttempts: 3,
    delay: 1000,
    backoff: 2, // exponential backoff multiplier
  },

  // Form validation
  validation: {
    passwordMinLength: 6,
    phonePattern: /^(0|\+84)[0-9]{9,10}$/,
    emailPattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  },

  // Working hours
  workingHours: {
    start: '07:00',
    end: '20:00',
    timeslotDuration: 20, // minutes
    breakStart: '12:00',
    breakEnd: '13:00',
  },

  // Appointment
  appointment: {
    cancelDeadlineHours: 24, // hours before appointment
    reminderHours: 24, // send reminder X hours before
    autoNoShowHours: 2, // mark as no-show after X hours
  },

  // Stock
  stock: {
    lowStockThreshold: 20,
    expiryWarningDays: 30,
  },

  // Report date ranges
  reportDateRanges: [
    { value: 'today', label: 'Hôm nay' },
    { value: 'yesterday', label: 'Hôm qua' },
    { value: 'thisWeek', label: 'Tuần này' },
    { value: 'lastWeek', label: 'Tuần trước' },
    { value: 'thisMonth', label: 'Tháng này' },
    { value: 'lastMonth', label: 'Tháng trước' },
    { value: 'last7Days', label: '7 ngày qua' },
    { value: 'last30Days', label: '30 ngày qua' },
    { value: 'custom', label: 'Tùy chỉnh' },
  ],

  // Export formats
  exportFormats: [
    { value: 'xlsx', label: 'Excel (.xlsx)' },
    { value: 'csv', label: 'CSV (.csv)' },
    { value: 'pdf', label: 'PDF (.pdf)' },
  ],
};

export default APP_CONFIG;