// frontend/src/routes/receptionistRoutes.jsx
import React, { Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { receptionistRoutes } from './index';
import DashboardLayout from '../components/layout/DashboardLayout';
import Loader from '../components/common/Loader';

export default function ReceptionistRoutes() {
  return (
    <DashboardLayout role="receptionist">
      <Suspense fallback={<Loader />}>
        <Routes>
          {receptionistRoutes.map((route) => (
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