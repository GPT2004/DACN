// frontend/src/components/auth/ProtectedRoute.jsx
import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import Loader from '../common/Loader';

/**
 * Lấy role dạng string an toàn từ user object
 */
const getRole = (user) => {
  if (!user) return null;
  if (typeof user.role === "string") return user.role;
  if (typeof user.role === "object") return user.role.name;
  return null;
};

const ProtectedRoute = ({ role, roles = [], children }) => {
  const { user, loading, isAuthenticated } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader size="large" />
      </div>
    );
  }

  if (!isAuthenticated || !user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  let userRole = getRole(user);
  // Normalize and sanitize role string (trim + lowercase)
  if (userRole && typeof userRole === 'string') {
    userRole = userRole.trim();
    // Normalize "Reception" to "receptionist" for route matching
    if (userRole.toLowerCase() === 'reception') userRole = 'receptionist';
  }
  const normalizedUserRole = userRole ? userRole.toLowerCase() : null;

  console.debug("USER ROLE (normalized):", normalizedUserRole, "REQUIRED ROLE:", role || roles);

  // Nếu route không yêu cầu role cụ thể → cho vào luôn
  if (!role && roles.length === 0) {
    return children;
  }

  // Check 1 role
  if (role && (!normalizedUserRole || normalizedUserRole !== String(role).trim().toLowerCase())) {
    return <Navigate to="/unauthorized" replace />;
  }

  // Check nhiều role
  if (roles.length > 0) {
    const allowed = roles.map(r => String(r).trim().toLowerCase());
    if (!normalizedUserRole || !allowed.includes(normalizedUserRole)) {
      return <Navigate to="/unauthorized" replace />;
    }
  }

  return children;
};

/**
 * Tự redirect người dùng về dashboard đúng role
 */
export const RoleBasedRedirect = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader size="large" />
      </div>
    );
  }

  let userRole = getRole(user);
  if (!userRole) return <Navigate to="/login" replace />;

  // Normalize "Reception" to "receptionist"
  if (userRole.toLowerCase() === 'reception') {
    userRole = 'receptionist';
  }

  const map = {
    admin: "/admin",
    doctor: "/doctor",
    patient: "/patient",
    receptionist: "/receptionist",
    pharmacist: "/pharmacist",
    labtech: "/labtech",
  };

  return <Navigate to={map[userRole.toLowerCase()] || "/login"} replace />;
};

/**
 * Route chỉ dành cho người chưa đăng nhập
 */
export const PublicRoute = ({ children }) => {
  const { user, loading } = useAuth();
  if (loading) return <Loader size="large" />;

  const userRole = getRole(user);
  const map = {
    admin: "/admin",
    doctor: "/doctor",
    patient: "/patient",
    receptionist: "/receptionist",
    pharmacist: "/pharmacist",
    labtech: "/labtech",
  };

  return userRole ? <Navigate to={map[userRole.toLowerCase()] || "/"} replace /> : children;
};

export default ProtectedRoute;
