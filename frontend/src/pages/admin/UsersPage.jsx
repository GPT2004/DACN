import React, { useEffect, useState } from 'react';
import AdminLayout from '../../components/admin/AdminLayout';
import { getAllUsers } from '../../services/userService';
import { ensureArray } from '../../utils/normalize';
import { Trash2, Edit2, Search } from 'lucide-react';
import EditUserModal from '../../components/admin/EditUserModal';
import DeleteUserModal from '../../components/admin/DeleteUserModal';
import AddUserModal from '../../components/admin/AddUserModal';
import UserDetailModal from '../../components/admin/UserDetailModal';

export default function UsersPage() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [editingUser, setEditingUser] = useState(null);
  const [deletingUser, setDeletingUser] = useState(null);
  const [addingUser, setAddingUser] = useState(false);
  const [detailUser, setDetailUser] = useState(null);

  useEffect(() => {
    loadUsers();
  }, []);

  const loadUsers = async () => {
    try {
      setLoading(true);
      // request a large limit to retrieve all users for scroll view
      const r = await getAllUsers({ limit: 10000 });
      const arr = ensureArray(r?.data?.users || r?.data || r);
      setUsers(arr);
    } catch (e) {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  const filteredUsers = users.filter(u =>
    (u.full_name || u.fullName || u.name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
    (u.email || '').toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getRoleColor = (role) => {
    const roleName = typeof role === 'object' ? role.name : role;
    const roleMap = {
      'ADMIN': 'bg-red-100 text-red-800',
      'DOCTOR': 'bg-blue-100 text-blue-800',
      'PATIENT': 'bg-green-100 text-green-800',
      'RECEPTIONIST': 'bg-purple-100 text-purple-800',
      'PHARMACIST': 'bg-orange-100 text-orange-800',
      'LAB_TECHNICIAN': 'bg-indigo-100 text-indigo-800',
    };
    return roleMap[roleName] || 'bg-gray-100 text-gray-800';
  };

  return (
    <AdminLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Quản lý người dùng</h1>
            <p className="text-gray-600 mt-1">Quản lý tài khoản và phân quyền người dùng hệ thống</p>
          </div>
          <button
            onClick={() => setAddingUser(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
          >
            + Thêm người dùng
          </button>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-4 top-3 text-gray-400" size={20} />
          <input
            type="text"
            placeholder="Tìm kiếm theo tên hoặc email..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
          />
        </div>

        {/* Table */}
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <div className="max-h-[60vh] overflow-y-auto">
              <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700 w-12">#</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Tên</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Email</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Vai trò</th>
                  <th className="px-6 py-4 text-center text-sm font-semibold text-gray-700 w-24">Hành động</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan="5" className="px-6 py-12 text-center">
                      <div className="flex justify-center">
                        <div className="spinner"></div>
                      </div>
                    </td>
                  </tr>
                ) : filteredUsers.length === 0 ? (
                  <tr>
                    <td colSpan="5" className="px-6 py-12 text-center text-gray-500">
                      Không tìm thấy người dùng nào
                    </td>
                  </tr>
                ) : (
                  filteredUsers.map((u, i) => (
                    <tr key={u.id} className="border-b border-gray-200 hover:bg-gray-50 transition cursor-pointer" onClick={() => setDetailUser(u)}>
                      <td className="px-6 py-4 text-sm text-gray-600">{i + 1}</td>
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">{u.full_name || u.fullName || u.name}</td>
                      <td className="px-6 py-4 text-sm text-gray-600">{u.email}</td>
                      <td className="px-6 py-4">
                        <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${getRoleColor(u.role)}`}>
                          {typeof u.role === 'object' ? u.role.name : u.role}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-center" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-center gap-2">
                          <button
                            onClick={() => setEditingUser(u)}
                            className="p-2 text-blue-600 hover:bg-blue-100 rounded-lg transition"
                            title="Chỉnh sửa"
                          >
                            <Edit2 size={18} />
                          </button>
                          <button
                            onClick={() => setDeletingUser(u)}
                            className="p-2 text-red-600 hover:bg-red-100 rounded-lg transition"
                            title="Xóa"
                          >
                            <Trash2 size={18} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
                </table>
              </div>
            </div>
          </div>

        {/* Footer Info */}
        {filteredUsers.length > 0 && (
          <div className="text-sm text-gray-600">
            Hiển thị <span className="font-semibold">{filteredUsers.length}</span> người dùng
          </div>
        )}
      </div>

      {/* Modals */}
      {editingUser && (
        <EditUserModal
          user={editingUser}
          onClose={() => setEditingUser(null)}
          onSuccess={loadUsers}
        />
      )}

      {deletingUser && (
        <DeleteUserModal
          user={deletingUser}
          onClose={() => setDeletingUser(null)}
          onSuccess={loadUsers}
        />
      )}
      {addingUser && (
        <AddUserModal
          onClose={() => setAddingUser(false)}
          onSuccess={() => {
            setAddingUser(false);
            loadUsers();
          }}
        />
      )}

      {detailUser && (
        <UserDetailModal
          user={detailUser}
          onClose={() => setDetailUser(null)}
        />
      )}
    </AdminLayout>
  );
}