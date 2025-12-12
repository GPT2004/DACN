import { ROLES } from './constants';

/**
 * Check if user has specific role
 */
export const hasRole = (user, role) => {
  if (!user || !user.role) return false;
  return user.role.name === role;
};

/**
 * Check if user has any of the specified roles
 */
export const hasAnyRole = (user, roles) => {
  if (!user || !user.role) return false;
  return roles.includes(user.role.name);
};

/**
 * Check if user is admin
 */
export const isAdmin = (user) => {
  return hasRole(user, ROLES.ADMIN);
};

/**
 * Check if user is doctor
 */
export const isDoctor = (user) => {
  return hasRole(user, ROLES.DOCTOR);
};

/**
 * Check if user is patient
 */
export const isPatient = (user) => {
  return hasRole(user, ROLES.PATIENT);
};

/**
 * Check if user is receptionist
 */
export const isReceptionist = (user) => {
  return hasRole(user, ROLES.RECEPTIONIST);
};

/**
 * Check if user is pharmacist
 */
export const isPharmacist = (user) => {
  return hasRole(user, ROLES.PHARMACIST);
};

/**
 * Check if user is lab tech
 */
export const isLabTech = (user) => {
  return hasRole(user, ROLES.LAB_TECH);
};

/**
 * Check if user can manage users
 */
export const canManageUsers = (user) => {
  return isAdmin(user);
};

/**
 * Check if user can manage appointments
 */
export const canManageAppointments = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.RECEPTIONIST, ROLES.DOCTOR]);
};

/**
 * Check if user can view medical records
 */
export const canViewMedicalRecords = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.DOCTOR, ROLES.PATIENT]);
};

/**
 * Check if user can create medical records
 */
export const canCreateMedicalRecords = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.DOCTOR]);
};

/**
 * Check if user can manage prescriptions
 */
export const canManagePrescriptions = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.DOCTOR, ROLES.PHARMACIST]);
};

/**
 * Check if user can dispense medicines
 */
export const canDispenseMedicines = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.PHARMACIST]);
};

/**
 * Check if user can manage stock
 */
export const canManageStock = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.PHARMACIST]);
};

/**
 * Check if user can view reports
 */
export const canViewReports = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.DOCTOR]);
};

/**
 * Check if user can manage lab orders
 */
export const canManageLabOrders = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.DOCTOR, ROLES.LAB_TECH]);
};

/**
 * Check if user can update lab results
 */
export const canUpdateLabResults = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.LAB_TECH]);
};

/**
 * Check if user can manage invoices
 */
export const canManageInvoices = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.RECEPTIONIST]);
};

/**
 * Check if user can use AI features
 */
export const canUseAIFeatures = (user) => {
  return hasAnyRole(user, [ROLES.ADMIN, ROLES.DOCTOR]);
};

/**
 * Get accessible routes based on role
 */
export const getAccessibleRoutes = (user) => {
  if (!user || !user.role) return [];

  const role = user.role.name;
  const routes = {
    [ROLES.ADMIN]: [
      '/admin',
      '/users',
      '/doctors',
      '/patients',
      '/appointments',
      '/medical-records',
      '/prescriptions',
      '/medicines',
      '/stocks',
      '/invoices',
      '/lab-orders',
      '/rooms',
      '/schedules',
      '/reports',
      '/audit-logs',
      '/settings',
    ],
    [ROLES.DOCTOR]: [
      '/doctor',
      '/appointments',
      '/medical-records',
      '/prescriptions',
      '/lab-orders',
      '/patients',
      '/schedules',
      '/ai-tools',
    ],
    [ROLES.PATIENT]: [
      '/patient',
      '/appointments',
      '/medical-records',
      '/prescriptions',
      '/lab-results',
      '/invoices',
      '/profile',
    ],
    [ROLES.RECEPTIONIST]: [
      '/receptionist',
      '/appointments',
      '/patients',
      '/check-in',
      '/invoices',
      '/schedules',
      '/rooms',
    ],
    [ROLES.PHARMACIST]: [
      '/pharmacist',
      '/prescriptions',
      '/medicines',
      '/stocks',
      '/dispensing',
    ],
    [ROLES.LAB_TECH]: [
      '/lab-tech',
      '/lab-orders',
      '/lab-results',
    ],
  };

  return routes[role] || [];
};

/**
 * Check if user can access route
 */
export const canAccessRoute = (user, route) => {
  const accessibleRoutes = getAccessibleRoutes(user);
  return accessibleRoutes.some(r => route.startsWith(r));
};