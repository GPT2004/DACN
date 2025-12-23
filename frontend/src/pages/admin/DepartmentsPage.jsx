/* eslint-disable no-console */
import React, { useEffect, useState } from 'react';
import AdminLayout from '../../components/admin/AdminLayout';
import { departmentService } from '../../services/departmentService';
import { Plus, Edit2, Trash2, Search, X } from 'lucide-react';

export default function DepartmentsPage() {
  const [departments, setDepartments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formData, setFormData] = useState({ name: '', slug: '', description: '', image_url: '' });

  useEffect(() => {
    loadDepartments();
  }, []);

  const loadDepartments = async () => {
    try {
      setLoading(true);
      const res = await departmentService.getAll();
      // Handle both array and object response
      const deptsData = Array.isArray(res) ? res : (res?.data?.data || res?.data || []);
      setDepartments(Array.isArray(deptsData) ? deptsData : []);
    } catch (e) {
      console.error('Failed to load departments', e);
      alert('Lỗi tải khoa: ' + (e?.message || 'Lỗi server'));
      setDepartments([]);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenModal = (dept = null) => {
    if (dept) {
      setEditingId(dept.id);
      setFormData({
        name: dept.name,
        slug: dept.slug || '',
        description: dept.description || '',
        image_url: dept.image_url || '',
      });
    } else {
      setEditingId(null);
      setFormData({ name: '', slug: '', description: '', image_url: '' });
    }
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setEditingId(null);
    setFormData({ name: '', slug: '', description: '', image_url: '' });
  };

  const handleSave = async () => {
    if (!formData.name.trim()) {
      alert('Tên khoa không được trống');
      return;
    }

    try {
      if (editingId) {
        await departmentService.update(editingId, formData);
        alert('Cập nhật khoa thành công');
      } else {
        await departmentService.create(formData);
        alert('Thêm khoa thành công');
      }
      handleCloseModal();
      loadDepartments();
    } catch (e) {
      alert('Lỗi: ' + (e?.message || 'Lỗi server'));
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc muốn xóa khoa này?')) return;

    try {
      await departmentService.delete(id);
      alert('Xóa khoa thành công');
      loadDepartments();
    } catch (e) {
      alert('Lỗi xóa: ' + (e?.message || 'Lỗi server'));
    }
  };

  const filtered = departments.filter(d =>
    d.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    d.slug?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <AdminLayout>
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Quản lý khoa</h1>
          <button
            onClick={() => handleOpenModal()}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            <Plus size={20} /> Thêm khoa
          </button>
        </div>

        <div className="mb-4 flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-3 text-gray-400" size={18} />
            <input
              type="text"
              placeholder="Tìm kiếm khoa..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {loading ? (
          <div className="text-center py-8">Đang tải...</div>
        ) : filtered.length ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">ID</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Tên khoa</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Slug</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Mô tả</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">Hành động</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(dept => (
                  <tr key={dept.id} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm text-gray-600">#{dept.id}</td>
                    <td className="py-3 px-4 text-sm font-medium text-gray-900">{dept.name}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">{dept.slug || '-'}</td>
                    <td className="py-3 px-4 text-sm text-gray-600 max-w-xs truncate">{dept.description || '-'}</td>
                    <td className="py-3 px-4 text-sm space-x-2">
                      <button
                        onClick={() => handleOpenModal(dept)}
                        className="text-blue-600 hover:text-blue-800 inline-flex items-center gap-1"
                        title="Sửa"
                      >
                        <Edit2 size={16} />
                      </button>
                      <button
                        onClick={() => handleDelete(dept.id)}
                        className="text-red-600 hover:text-red-800 inline-flex items-center gap-1"
                        title="Xóa"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">Không có khoa nào</div>
        )}

        {/* Modal */}
        {showModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full">
              <div className="flex justify-between items-center p-6 border-b">
                <h2 className="text-xl font-bold">{editingId ? 'Sửa khoa' : 'Thêm khoa'}</h2>
                <button onClick={handleCloseModal} className="text-gray-500 hover:text-gray-700">
                  <X size={24} />
                </button>
              </div>

              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Tên khoa *</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="VD: Tim mạch"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Slug</label>
                  <input
                    type="text"
                    value={formData.slug}
                    onChange={(e) => setFormData({ ...formData, slug: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="VD: tim-mach"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Mô tả</label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Mô tả khoa"
                    rows="3"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">URL hình ảnh</label>
                  <input
                    type="text"
                    value={formData.image_url}
                    onChange={(e) => setFormData({ ...formData, image_url: e.target.value })}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="https://example.com/image.jpg"
                  />
                </div>
              </div>

              <div className="flex gap-3 p-6 border-t">
                <button
                  onClick={handleCloseModal}
                  className="flex-1 px-4 py-2 border rounded-lg text-gray-700 hover:bg-gray-50"
                >
                  Hủy
                </button>
                <button
                  onClick={handleSave}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  {editingId ? 'Cập nhật' : 'Thêm'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </AdminLayout>
  );
}

