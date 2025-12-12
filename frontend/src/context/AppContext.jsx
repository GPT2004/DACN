// frontend/src/context/AppContext.jsx
import React, { createContext, useState, useContext } from 'react';

const AppContext = createContext(null);

export const AppProvider = ({ children }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [breadcrumbs, setBreadcrumbs] = useState([]);

  const toggleSidebar = () => {
    setSidebarCollapsed(prev => !prev);
  };

  const updateBreadcrumbs = (items) => {
    setBreadcrumbs(items);
  };

  const value = {
    sidebarCollapsed,
    setSidebarCollapsed,
    toggleSidebar,
    breadcrumbs,
    updateBreadcrumbs,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
};