import React, { useEffect, useState } from 'react';
import AdminLayout from '../../components/admin/AdminLayout';
import { getAllDoctors } from '../../services/doctorService';
import { ensureArray } from '../../utils/normalize';
import { Trash2, Edit2, Search, Stethoscope } from 'lucide-react';
import EditDoctorModal from '../../components/admin/EditDoctorModal';
import DeleteDoctorModal from '../../components/admin/DeleteDoctorModal';
import AddDoctorModal from '../../components/admin/AddDoctorModal';
import DoctorDetailModal from '../../components/admin/DoctorDetailModal';
import ImageViewer from '../../components/common/ImageViewer';

export default function DoctorsPage() {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [editingDoctor, setEditingDoctor] = useState(null);
  const [deletingDoctor, setDeletingDoctor] = useState(null);
  const [addingDoctor, setAddingDoctor] = useState(false);
  const [detailDoctor, setDetailDoctor] = useState(null);
  const [viewerUrl, setViewerUrl] = useState('');

  useEffect(() => {
    loadDoctors();
  }, []);

  const loadDoctors = async () => {
    try {
      setLoading(true);
      // request many results so admin can scroll through all doctors
      const response = await getAllDoctors({ limit: 10000 });
      const arr = ensureArray(response?.data?.doctors || response?.data || response);
      setDoctors(arr);
    } catch (e) {
      console.error('Error loading doctors:', e);
    } finally {
      setLoading(false);
    }
  };

  const filtered = (doctors || []).filter(d =>
    (d.user?.full_name || d.full_name || d.fullName || d.name || '').toLowerCase().includes(query.toLowerCase()) ||
    (d.user?.email || d.email || '').toLowerCase().includes(query.toLowerCase())
  );

 

  return (
    <AdminLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Stethoscope className="text-blue-600" size={32} />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Quản lý bác sĩ</h1>
                <p className="text-gray-600 mt-1">Quản lý thông tin và lịch khám của bác sĩ</p>
              </div>
            </div>
          </div>
          <button
            onClick={() => setAddingDoctor(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
          >
            + Thêm bác sĩ
          </button>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-4 top-3 text-gray-400" size={20} />
          <input
            type="text"
            placeholder="Tìm kiếm theo tên hoặc email..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
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
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Tên bác sĩ</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Email</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Chuyên môn</th>
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
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan="5" className="px-6 py-12 text-center text-gray-500">
                      Không tìm thấy bác sĩ nào
                    </td>
                  </tr>
                ) : (
                  filtered.map((d, i) => (
                    <tr key={d.id} className="border-b border-gray-200 hover:bg-gray-50 transition cursor-pointer" onClick={() => setDetailDoctor(d)}>
                      <td className="px-6 py-4 text-sm text-gray-600">{i + 1}</td>
                      <td className="px-6 py-4 text-sm font-medium text-gray-900 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full overflow-hidden bg-gray-100">
                          {d.user?.avatar_url ? (
                            <img src={d.user.avatar_url} alt="avatar" className="w-full h-full object-cover cursor-pointer" onClick={(e) => { e.stopPropagation(); setViewerUrl(d.user.avatar_url); }} />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-gray-400">N/A</div>
                          )}
                        </div>
                        <div>{d.user?.full_name || d.full_name || d.fullName || d.name}</div>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-600">{d.user?.email || d.email}</td>
                      <td className="px-6 py-4 text-sm">
                        <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full font-medium">
                          {d.specialties && Array.isArray(d.specialties) && d.specialties.length > 0 
                            ? d.specialties.join(', ') 
                            : d.specialty || 'Chưa cập nhật'}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-center" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-center gap-2">
                          <button
                            onClick={() => setEditingDoctor(d)}
                            className="p-2 text-blue-600 hover:bg-blue-100 rounded-lg transition"
                            title="Chỉnh sửa"
                          >
                            <Edit2 size={18} />
                          </button>
                          <button
                            onClick={() => setDeletingDoctor(d)}
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
        {filtered.length > 0 && (
          <div className="text-sm text-gray-600">
            Hiển thị <span className="font-semibold">{filtered.length}</span> bác sĩ
          </div>
        )}
      </div>

      {/* Modals */}
      {editingDoctor && (
        <EditDoctorModal
          doctor={editingDoctor}
          onClose={() => setEditingDoctor(null)}
          onSuccess={loadDoctors}
        />
      )}

      {deletingDoctor && (
        <DeleteDoctorModal
          doctor={deletingDoctor}
          onClose={() => setDeletingDoctor(null)}
          onSuccess={loadDoctors}
        />
      )}
      {addingDoctor && (
        <AddDoctorModal
          onClose={() => setAddingDoctor(false)}
          onSuccess={() => {
            setAddingDoctor(false);
            loadDoctors();
          }}
        />
      )}

      {detailDoctor && (
        <DoctorDetailModal
          doctor={detailDoctor}
          onClose={() => setDetailDoctor(null)}
        />
      )}
      {viewerUrl && (
        <ImageViewer src={viewerUrl} onClose={() => setViewerUrl('')} />
      )}
    </AdminLayout>
  );
}
