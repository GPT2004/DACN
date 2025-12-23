/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-console */
/* eslint-disable no-unused-vars */
// frontend/src/components/admin/UserManagement/UserList.jsx
import React, { useState, useEffect } from 'react';
import { Search, Plus, Edit, Trash2, Eye, ToggleLeft, ToggleRight, Key } from 'lucide-react';
import { userService } from '../../../services/userService';
import { formatDate } from '../../../utils/helpers';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Badge from '../../common/Badge';
import Loader from '../../common/Loader';
import Pagination from '../../common/Pagination';

export default function UserList({ onEdit, onCreate, onView }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [roleFilter, setRoleFilter] = useState('');

  useEffect(() => {
    fetchUsers();
  }, [page, searchTerm, roleFilter]);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      const response = await userService.getAllUsers({
        page,
        limit: 10,
        search: searchTerm,
        role: roleFilter
      });
      setUsers(response.data.users || []);
      setTotalPages(response.data.pagination?.totalPages || 1);
    } catch (error) {
      console.error('Fetch users error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleStatus = async (id, currentStatus) => {
    try {
      await userService.toggleUserStatus(id, !currentStatus);
      fetchUsers();
    } catch (error) {
      console.error('Toggle status error:', error);
    }
  };

  const handleResetPassword = async (id) => {
    const newPassword = prompt('Nhập mật khẩu mới:');
    if (!newPassword) return;

    try {
      await userService.resetUserPassword(id, newPassword);
      alert('Đặt lại mật khẩu thành công!');
    } catch (error) {
      console.error('Reset password error:', error);
      alert('Có lỗi xảy ra!');
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa người dùng này?')) return;
    
    try {
      await userService.deleteUser(id);
      fetchUsers();
    } catch (error) {
      console.error('Delete user error:', error);
      alert('Không thể xóa người dùng này!');
    }
  };

  const getRoleBadge = (role) => {
    const roleColors = {
      Admin: 'bg-purple-100 text-purple-800',
      Doctor: 'bg-blue-100 text-blue-800',
      Patient: 'bg-green-100 text-green-800',
      Receptionist: 'bg-yellow-100 text-yellow-800',
      Pharmacist: 'bg-pink-100 text-pink-800',
      LabTech: 'bg-indigo-100 text-indigo-800',
    };

    return (
      <span className={`px-2 py-1 text-xs font-medium rounded-full ${roleColors[role] || 'bg-gray-100 text-gray-800'}`}>
        {role}
      </span>
    );
  };

  if (loading) {
    return <Loader />;
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Quản lý Người dùng</h2>
        <Button onClick={onCreate}>
          <Plus className="w-4 h-4 mr-2" />
          Thêm Người dùng
        </Button>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="flex-1">
          <Input
            placeholder="Tìm kiếm theo tên, email, số điện thoại..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <select
          className="px-4 py-2 border border-gray-300 rounded-lg"
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
        >
          <option value="">Tất cả vai trò</option>
          <option value="Admin">Admin</option>
          <option value="Doctor">Bác sĩ</option>
          <option value="Patient">Bệnh nhân</option>
          <option value="Receptionist">Lễ tân</option>
          <option value="Pharmacist">Dược sĩ</option>
          <option value="LabTech">Kỹ thuật viên</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Người dùng
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Vai trò
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Liên hệ
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Ngày tạo
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Trạng thái
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                Thao tác
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {users.map((user) => (
              <tr key={user.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <img
                      className="h-10 w-10 rounded-full"
                      src={user.avatar_url || '/avatar-placeholder.png'}
                      alt={user.full_name}
                    />
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900">
                        {user.full_name}
                      </div>
                      <div className="text-sm text-gray-500">
                        {user.email}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getRoleBadge(user.role?.name)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {user.phone}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {formatDate(user.created_at)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <Badge 
                    text={user.is_active ? 'Hoạt động' : 'Ngừng hoạt động'}
                  />
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => onView(user)}
                      className="text-blue-600 hover:text-blue-900"
                      title="Xem chi tiết"
                    >
                      <Eye className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => onEdit(user)}
                      className="text-yellow-600 hover:text-yellow-900"
                      title="Chỉnh sửa"
                    >
                      <Edit className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => handleResetPassword(user.id)}
                      className="text-purple-600 hover:text-purple-900"
                      title="Đặt lại mật khẩu"
                    >
                      <Key className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => handleToggleStatus(user.id, user.is_active)}
                      className="text-green-600 hover:text-green-900"
                      title="Đổi trạng thái"
                    >
                      {user.is_active ? (
                        <ToggleRight className="w-5 h-5" />
                      ) : (
                        <ToggleLeft className="w-5 h-5" />
                      )}
                    </button>
                    <button
                      onClick={() => handleDelete(user.id)}
                      className="text-red-600 hover:text-red-900"
                      title="Xóa"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {users.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">Không có dữ liệu</p>
          </div>
        )}
      </div>

      {/* Pagination */}
      <Pagination
        currentPage={page}
        totalPages={totalPages}
        onPageChange={setPage}
      />
    </div>
  );
}