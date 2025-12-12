// frontend/src/config/api.config.js
/**
 * API Configuration
 * Centralized API endpoint và timeout configuration
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080/api';
const AI_SERVICE_URL = process.env.REACT_APP_AI_SERVICE_URL || 'http://localhost:5000';

export const API_CONFIG = {
  // Base URLs
  baseURL: API_BASE_URL,
  aiServiceURL: AI_SERVICE_URL,

  // Timeout settings (ms)
  timeout: 30000,
  
  // Retry configuration
  retry: {
    attempts: 3,
    delay: 1000, // 1 second
  },

  // Endpoints
  endpoints: {
    // Auth
    auth: {
      login: '/auth/login',
      register: '/auth/register',
      logout: '/auth/logout',
      profile: '/auth/me',
      changePassword: '/auth/change-password',
      refreshToken: '/auth/refresh',
    },

    // Users
    users: {
      base: '/users',
      stats: '/users/stats',
      byRole: (role) => `/users/role/${role}`,
      profile: '/users/me/profile',
      changePassword: '/users/me/change-password',
    },

    // Patients
    patients: {
      base: '/patients',
      profile: '/patients/me/profile',
      appointments: (id) => `/patients/${id}/appointments`,
      medicalRecords: (id) => `/patients/${id}/medical-records`,
      prescriptions: (id) => `/patients/${id}/prescriptions`,
      invoices: (id) => `/patients/${id}/invoices`,
    },

    // Doctors
    doctors: {
      base: '/doctors',
      public: '/doctors/public',
      specialties: '/doctors/public/specialties/list',
      appointments: (id) => `/doctors/${id}/appointments`,
      schedules: (id) => `/doctors/${id}/schedules`,
      patients: (id) => `/doctors/${id}/patients`,
      stats: (id) => `/doctors/${id}/stats`,
    },

    // Appointments
    appointments: {
      base: '/appointments',
      confirm: (id) => `/appointments/${id}/confirm`,
      checkIn: (id) => `/appointments/${id}/check-in`,
      start: (id) => `/appointments/${id}/start`,
      complete: (id) => `/appointments/${id}/complete`,
      cancel: (id) => `/appointments/${id}/cancel`,
      noShow: (id) => `/appointments/${id}/no-show`,
    },

    // Medical Records
    medicalRecords: {
      base: '/medical-records',
      patientHistory: (id) => `/medical-records/patient/${id}/history`,
      attachments: (id) => `/medical-records/${id}/attachments`,
    },

    // Prescriptions
    prescriptions: {
      base: '/prescriptions',
      approve: (id) => `/prescriptions/${id}/approve`,
      dispense: (id) => `/prescriptions/${id}/dispense`,
      download: (id) => `/prescriptions/${id}/pdf`,
    },

    // Medicines
    medicines: {
      base: '/medicines',
      stocks: '/medicines/stocks/all',
      summary: '/medicines/stocks/summary',
      lowStock: '/medicines/stocks/low-stock',
      expiring: '/medicines/stocks/expiring',
    },

    // Invoices
    invoices: {
      base: '/invoices',
      revenue: '/invoices/reports/revenue',
      pay: (id) => `/invoices/${id}/pay`,
      refund: (id) => `/invoices/${id}/refund`,
      download: (id) => `/invoices/${id}/pdf`,
    },

    // Lab Orders
    labOrders: {
      base: '/lab-orders',
      results: (id) => `/lab-orders/${id}/results`,
      complete: (id) => `/lab-orders/${id}/complete`,
    },

    // Schedules & Timeslots
    schedules: {
      base: '/schedules',
      timeslots: '/schedules/timeslots',
      available: '/schedules/timeslots/available',
    },

    timeslots: {
      base: '/timeslots',
      available: '/timeslots/available',
      doctor: (id) => `/timeslots/doctor/${id}`,
      appointments: (id) => `/timeslots/${id}/appointments`,
      bulk: '/timeslots/bulk',
    },

    // Rooms
    rooms: {
      base: '/rooms',
      available: '/rooms/available',
      schedule: (id) => `/rooms/${id}/schedule`,
      status: (id) => `/rooms/${id}/status`,
    },

    // Notifications
    notifications: {
      base: '/notifications',
      unreadCount: '/notifications/unread/count',
      markAsRead: (id) => `/notifications/${id}/read`,
      markAllAsRead: '/notifications/read-all',
      send: '/notifications/send',
      broadcast: '/notifications/broadcast',
    },

    // Reports
    reports: {
      dashboard: '/reports/dashboard',
      revenue: '/reports/revenue',
      appointments: '/reports/appointments',
      doctorPerformance: '/reports/doctors/performance',
      commonDiseases: '/reports/diseases/common',
      stock: '/reports/stock',
      patients: '/reports/patients',
      export: '/reports/export',
    },

    // Audit Logs
    auditLogs: {
      base: '/audit_logs',
      user: (userId) => `/audit_logs/user/${userId}`,
      actionTypes: '/audit_logs/actions/types',
      cleanup: '/audit_logs/cleanup',
    },

    // AI Services
    ai: {
      symptomChecker: `${AI_SERVICE_URL}/symptom-checker`,
      riskPrediction: `${AI_SERVICE_URL}/risk-prediction`,
      analyzeImage: `${AI_SERVICE_URL}/analyze-image`,
    },
  },

  // HTTP Status codes
  statusCodes: {
    OK: 200,
    CREATED: 201,
    NO_CONTENT: 204,
    BAD_REQUEST: 400,
    UNAUTHORIZED: 401,
    FORBIDDEN: 403,
    NOT_FOUND: 404,
    CONFLICT: 409,
    UNPROCESSABLE_ENTITY: 422,
    INTERNAL_SERVER_ERROR: 500,
    SERVICE_UNAVAILABLE: 503,
  },

  // Request headers
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },

  // File upload settings
  upload: {
    maxSize: 5 * 1024 * 1024, // 5MB
    allowedTypes: {
      images: ['image/jpeg', 'image/jpg', 'image/png'],
      documents: ['application/pdf', 'image/jpeg', 'image/png'],
    },
  },
};

export default API_CONFIG;