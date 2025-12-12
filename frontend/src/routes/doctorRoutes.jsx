// frontend/src/routes/doctorRoutes.jsx
import React, { Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { doctorRoutes } from './index';
import DashboardLayout from '../components/layout/DashboardLayout';
import Loader from '../components/common/Loader';

export default function DoctorRoutes() {
  return (
    <DashboardLayout role="doctor">
      <Suspense fallback={<Loader />}>
        <Routes>
          {doctorRoutes.map((route) => (
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