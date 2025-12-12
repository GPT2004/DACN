import React from "react";
import { Link } from "react-router-dom";

export default function UnauthorizedPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-yellow-50 p-6">
      <h1 className="text-4xl font-bold text-yellow-700 mb-4">403 - Không có quyền truy cập</h1>
      <p className="text-lg text-gray-700 mb-6">
        Bạn không có quyền truy cập vào trang này. Vui lòng đăng nhập đúng tài khoản hoặc liên hệ quản trị viên.
      </p>

      <div className="flex gap-4">
        <Link
          to="/login"
          className="px-5 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
        >
          Đăng nhập
        </Link>

        <Link
          to="/"
          className="px-5 py-2 bg-gray-300 rounded hover:bg-gray-400 transition"
        >
          Về trang chủ
        </Link>
      </div>
    </div>
  );
}
