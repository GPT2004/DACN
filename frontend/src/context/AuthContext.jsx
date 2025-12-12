  // frontend/src/context/AuthContext.jsx
  /**
   * Auth Context - Quản lý toàn bộ authentication state
   * FIXED: role luôn là string (ví dụ: "Admin")
   */

  import React, { createContext, useState, useEffect, useContext, useCallback } from 'react';
  import { authService } from '../services/authService';
  import { 
    getAuthToken, 
    setAuthToken, 
    setRefreshToken,
    setUserData,
    clearAuthData,
    getUserData 
  } from '../utils/storage';

  // Create Context
  export const AuthContext = createContext(null);

  // Provider Component
  export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    // === HÀM CHUẨN HÓA ROLE ===
    const normalizeRole = (u) => {
      if (!u || !u.role) {
        u.role = 'Patient'; // fallback
        return u;
      }
      if (typeof u.role === 'object') {
        u.role = u.role.name || u.role.role || 'Patient';
      }
      // Keep role as-is, don't lowercase here
      return u;
    };

    // Initialize auth state from storage
    useEffect(() => {
      const initAuth = async () => {
        try {
          const token = getAuthToken();
          const savedUser = getUserData();

          if (token && savedUser) {
                try {
                  const response = await authService.getProfile();
                  // unwrap common response shapes: { data: { ...user } } or { data: { user: {...} } }
                  let freshUser = response?.data?.data || response?.data?.user || response?.data || null;
                  freshUser = normalizeRole(freshUser); // CHUẨN HÓA ROLE

                  setUser(freshUser);
                  setUserData(freshUser);
                  setIsAuthenticated(true);
                } catch (err) {
                  clearAuthData();
                  setUser(null);
                  setIsAuthenticated(false);
                }
          }
        } catch (err) {
          console.error('Auth initialization error:', err);
          clearAuthData();
        } finally {
          setLoading(false);
        }
      };

      initAuth();
    }, []);

    // Login
    const login = useCallback(async (email, password) => {
    try {
      setLoading(true);
      setError(null);

      const response = await authService.login(email, password);

      // CHỈ UNWRAP 1 LẦN
      let payload = response;
      if (payload && typeof payload === 'object' && 'data' in payload) {
        payload = payload.data;
      }

      console.log("LOGIN PAYLOAD:", payload); // DEBUG

      // LẤY ĐÚNG user từ payload.data.user
      const userData = payload?.data?.user || payload?.user || null;
      const token = payload?.data?.token || payload?.token || null;
      const refreshToken = payload?.data?.refreshToken || payload?.refreshToken || null;

      if (!userData) {
        const msg = payload?.message || 'Không nhận được thông tin user';
        setError(msg);
        setLoading(false);
        return { success: false, error: msg };
      }

      // CHUẨN HÓA ROLE THÀNH STRING
      if (userData.role && typeof userData.role === 'object') {
        userData.role = userData.role.name || 'Patient';
      }
      // Keep original role casing, backend will normalize

      // LƯU VÀO STORAGE
      if (token) setAuthToken(token);
      if (refreshToken) setRefreshToken(refreshToken);
      setUserData(userData); // ← BÂY GIỜ SẼ LƯU ĐÚNG!

      // CẬP NHẬT CONTEXT
      setUser(userData);
      setIsAuthenticated(true);

      return { success: true, user: userData };
    } catch (err) {
      const errorMsg = err.response?.data?.message || err.message || 'Đăng nhập thất bại';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setLoading(false);
    }
  }, []);

    // Register
    const register = useCallback(async (userData) => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await authService.register(userData);
        const { user: newUser, token, refreshToken } = response.data;

        const normalizedUser = normalizeRole(newUser);

        setAuthToken(token);
        setRefreshToken(refreshToken);
        setUserData(normalizedUser);

        setUser(normalizedUser);
        setIsAuthenticated(true);

        return { success: true, user: normalizedUser };
      } catch (err) {
        const errorMsg = err.response?.data?.message || 'Đăng ký thất bại';
        setError(errorMsg);
        return { success: false, error: errorMsg };
      } finally {
        setLoading(false);
      }
    }, []);

    // Logout
    const logout = useCallback(async () => {
      try {
        await authService.logout();
      } catch (err) {
        console.error('Logout error:', err);
      } finally {
        clearAuthData();
        setUser(null);
        setIsAuthenticated(false);
        setError(null);
      }
    }, []);

    // Update profile
    const updateProfile = useCallback(async () => {
      try {
        const response = await authService.getProfile();
        let updatedUser = response?.data?.data || response?.data?.user || response?.data || null;
        updatedUser = normalizeRole(updatedUser);

        setUserData(updatedUser);
        setUser(updatedUser);
        
        return { success: true, user: updatedUser };
      } catch (err) {
        return { success: false, error: err.message };
      }
    }, []);

    // Refresh user data
    const refreshUser = useCallback(async () => {
      try {
        const response = await authService.getProfile();
        let freshUser = response?.data?.data || response?.data?.user || response?.data || null;
        freshUser = normalizeRole(freshUser);

        setUserData(freshUser);
        setUser(freshUser);
        
        return freshUser;
      } catch (err) {
        console.error('Refresh user error:', err);
        return null;
      }
    }, []);

    // Change password
    const changePassword = useCallback(async (currentPassword, newPassword) => {
      try {
        await authService.changePassword(currentPassword, newPassword);
        return { success: true };
      } catch (err) {
        const errorMsg = err.response?.data?.message || 'Đổi mật khẩu thất bại';
        return { success: false, error: errorMsg };
      }
    }, []);

    // === ROLE HELPERS ===
    const getRoleName = (u) => {
      if (!u || !u.role) return null;
      return typeof u.role === 'string' ? u.role : null;
    };

    const hasRole = useCallback((role) => {
      const rn = getRoleName(user);
      return rn ? rn.toLowerCase() === role.toLowerCase() : false;
    }, [user]);

    const hasAnyRole = useCallback((roles) => {
      const rn = getRoleName(user);
      return rn ? roles.map(r => r.toLowerCase()).includes(rn.toLowerCase()) : false;
    }, [user]);

    // === PERMISSIONS ===
    const can = useCallback((permission) => {
      if (!user || !user.role) return false;
      
      const rolePermissions = {
        Admin: ['*'],
        Doctor: ['view:patients', 'create:medical-records', 'create:prescriptions', 'view:appointments'],
        Receptionist: ['create:appointments', 'view:patients', 'manage:invoices'],
        Patient: ['view:own-data', 'book:appointments'],
        Pharmacist: ['dispense:medicines', 'manage:stock'],
        LabTech: ['manage:lab-orders', 'update:lab-results'],
      };

      const userRoleStr = typeof user.role === 'string' ? user.role : '';
      const userPermissions = rolePermissions[userRoleStr] || [];
      
      if (userPermissions.includes('*')) return true;
      return userPermissions.includes(permission);
    }, [user]);

    // === GETTERS ===
    const getUserName = useCallback(() => user?.full_name || 'User', [user]);
    const getUserRole = useCallback(() => getRoleName(user), [user]);
    const getUserAvatar = useCallback(() => user?.avatar_url || null, [user]);

    // === CONTEXT VALUE ===
    const value = {
      user,
      loading,
      error,
      isAuthenticated,
      login,
      register,
      logout,
      updateProfile,
      refreshUser,
      changePassword,
      hasRole,
      hasAnyRole,
      can,
      getUserName,
      getUserRole,
      getUserAvatar,
      setUser,
      setError,
    };

    return (
      <AuthContext.Provider value={value}>
        {children}
      </AuthContext.Provider>
    );
  };

  // Custom hook
  export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
      throw new Error('useAuth must be used within AuthProvider');
    }
    return context;
  };

  export default AuthContext;