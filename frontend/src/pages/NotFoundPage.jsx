import React from "react";
import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-6">
      <h1 className="text-6xl font-bold text-gray-700 mb-4">404</h1>
      <p className="text-lg text-gray-600 mb-6">
        Trang bạn tìm kiếm không tồn tại hoặc đã bị xóa.
      </p>

      <Link
        to="/"
        className="px-5 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
      >
        Quay lại trang chủ
      </Link>
    </div>
  );
}
