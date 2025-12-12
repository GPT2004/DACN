// frontend/src/routes/patientRoutes.jsx
import React, { Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { patientRoutes } from './index';
import DashboardLayout from '../components/layout/DashboardLayout';
import Loader from '../components/common/Loader';

export default function PatientRoutes() {
  return (
    <DashboardLayout role="patient">
      <Suspense fallback={<Loader />}>
        <Routes>
          {patientRoutes.map((route) => (
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
