import React, { useState, useEffect } from 'react';
import { Edit2, Trash2, Search, X, BookOpen } from 'lucide-react';
import specialtyService from '../../services/specialtyService';
import AddSpecialtyModal from './AddSpecialtyModal';
import EditSpecialtyModal from './EditSpecialtyModal';
import DeleteSpecialtyModal from './DeleteSpecialtyModal';
import SpecialtyDetailModal from './SpecialtyDetailModal';

export default function SpecialtiesModal({ isOpen, onClose }) {
  const [specialties, setSpecialties] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [addingSpecialty, setAddingSpecialty] = useState(false);
  const [editingSpecialty, setEditingSpecialty] = useState(null);
  const [deletingSpecialty, setDeletingSpecialty] = useState(null);
  const [detailSpecialty, setDetailSpecialty] = useState(null);

  useEffect(() => {
    if (isOpen) {
      loadSpecialties();
    }
  }, [isOpen]);

  const loadSpecialties = async () => {
    setLoading(true);
    try {
      const response = await specialtyService.getAll();
      const data = Array.isArray(response) ? response : response?.data || [];
      setSpecialties(data);
    } catch (error) {
      console.error('Failed to load specialties:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredSpecialties = specialties.filter(s =>
    s.name.toLowerCase().includes(search.toLowerCase()) ||
    (s.slug && s.slug.toLowerCase().includes(search.toLowerCase()))
  );

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50">
      {/* Background Overlay */}
      <div 
        className="absolute inset-0 bg-black bg-opacity-30"
        onClick={onClose}
      />
      
      {/* Sliding Panel from Right */}
      <div className="absolute right-0 top-0 h-full w-full md:w-2xl bg-white shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b shrink-0">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Quản lý chuyên khoa</h1>
            <p className="text-gray-600 mt-1">Quản lý danh sách các chuyên khoa của phòng khám</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-900 transition"
            disabled={loading}
          >
            <X size={28} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-4 top-3 text-gray-400" size={20} />
              <input
                type="text"
                placeholder="Tìm kiếm theo tên hoặc email..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
              />
            </div>

            {/* Add Button */}
            <div className="flex justify-end">
              <button
                onClick={() => setAddingSpecialty(true)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
                disabled={loading}
              >
                + Thêm chuyên khoa
              </button>
            </div>

            {/* Table */}
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700 w-12">#</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Tên Chuyên Khoa</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Slug</th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Mô Tả</th>
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
                    ) : filteredSpecialties.length === 0 ? (
                      <tr>
                        <td colSpan="5" className="px-6 py-12 text-center text-gray-500">
                          Không tìm thấy khoa nào
                        </td>
                      </tr>
                    ) : (
                      filteredSpecialties.map((s, i) => (
                        <tr key={s.id} className="border-b border-gray-200 hover:bg-gray-50 transition cursor-pointer" onClick={() => setDetailSpecialty(s)}>
                          <td className="px-6 py-4 text-sm text-gray-600">{i + 1}</td>
                          <td className="px-6 py-4 text-sm font-medium text-gray-900">{s.name}</td>
                          <td className="px-6 py-4 text-sm text-gray-600">{s.slug || '-'}</td>
                          <td className="px-6 py-4 text-sm text-gray-600 truncate max-w-xs">{s.description || '-'}</td>
                          <td className="px-6 py-4 text-center" onClick={(e) => e.stopPropagation()}>
                            <div className="flex items-center justify-center gap-2">
                              <button
                                onClick={() => setEditingSpecialty(s)}
                                className="p-2 text-blue-600 hover:bg-blue-100 rounded-lg transition"
                                title="Chỉnh sửa"
                              >
                                <Edit2 size={18} />
                              </button>
                              <button
                                onClick={() => setDeletingSpecialty(s)}
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

            {/* Footer Info */}
            {filteredSpecialties.length > 0 && (
              <div className="text-sm text-gray-600">
                Hiển thị <span className="font-semibold">{filteredSpecialties.length}</span> chuyên khoa
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Modals */}
      {addingSpecialty && (
        <AddSpecialtyModal
          onClose={() => setAddingSpecialty(false)}
          onSuccess={loadSpecialties}
        />
      )}

      {editingSpecialty && (
        <EditSpecialtyModal
          specialty={editingSpecialty}
          onClose={() => setEditingSpecialty(null)}
          onSuccess={loadSpecialties}
        />
      )}

      {deletingSpecialty && (
        <DeleteSpecialtyModal
          specialty={deletingSpecialty}
          onClose={() => setDeletingSpecialty(null)}
          onSuccess={loadSpecialties}
        />
      )}

      {detailSpecialty && (
        <SpecialtyDetailModal
          specialty={detailSpecialty}
          onClose={() => setDetailSpecialty(null)}
        />
      )}
    </div>
  );
}
