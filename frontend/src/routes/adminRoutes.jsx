// frontend/src/routes/adminRoutes.jsx
import React, { Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { adminRoutes } from './index';
import DashboardLayout from '../components/layout/DashboardLayout';
import Loader from '../components/common/Loader';

export default function AdminRoutes() {
  return (
    <DashboardLayout role="admin">
      <Suspense fallback={<Loader />}>
        <Routes>
          {adminRoutes.map((route) => (
            <Route
              key={route.path}
              path={route.path}
              element={<route.element />}
            />
          ))}
        </Routes>
      </Suspense>
    </DashboardLayout>
  );
}