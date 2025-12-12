import React, { useEffect, useState, useRef } from 'react';
import ImageViewer from '../../components/common/ImageViewer';
import AdminLayout from '../../components/admin/AdminLayout';
import specialtyService from '../../services/specialtyService';
import { Plus, Edit2, Trash2, Search, X } from 'lucide-react';

export default function SpecialtyPage() {
  const [specialties, setSpecialties] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formData, setFormData] = useState({ name: '', slug: '', description: '', image_url: '' });
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState('');
  const fileRef = useRef(null);
  const [showViewer, setShowViewer] = useState(false);
  const [viewerSrc, setViewerSrc] = useState('');

  // cleanup preview blob url on unmount / when preview changes
  useEffect(() => {
    return () => {
      try {
        if (imagePreview && imagePreview.startsWith && imagePreview.startsWith('blob:')) {
          URL.revokeObjectURL(imagePreview);
        }
      } catch (err) {}
    };
  }, [imagePreview]);

  useEffect(() => {
    loadSpecialties();
  }, []);

  const loadSpecialties = async () => {
    try {
      setLoading(true);
      // request many specialties so admin can scroll through the full list
      const res = await specialtyService.getAll({ limit: 10000 });
      // Normalize different response shapes:
      // - service may return payload.data (i.e. { specialties: [...] })
      // - or payload (i.e. { success, message, data: { specialties: [...] } })
      // - or directly an array
      const specialtiesData = Array.isArray(res)
        ? res
        : (res?.specialties || res?.data?.specialties || res?.data?.data?.specialties || []);
      setSpecialties(Array.isArray(specialtiesData) ? specialtiesData : []);
    } catch (e) {
      alert('Lỗi tải chuyên khoa: ' + (e?.message || 'Lỗi server'));
      setSpecialties([]);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenModal = (specialty = null) => {
    if (specialty) {
      setEditingId(specialty.id);
      setFormData({
        name: specialty.name,
        slug: specialty.slug || '',
        description: specialty.description || '',
        image_url: specialty.image_url || '',
      });
      setImagePreview(specialty.image_url || '');
      setImageFile(null);
    } else {
      setEditingId(null);
      setFormData({ name: '', slug: '', description: '', image_url: '' });
      setImageFile(null);
      setImagePreview('');
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
      alert('Tên chuyên khoa không được trống');
      return;
    }

    try {
      // If an image file is selected, submit as multipart/form-data
      if (imageFile) {
        const fd = new FormData();
        fd.append('image_url', imageFile);
        fd.append('name', formData.name);
        if (formData.slug) fd.append('slug', formData.slug);
        if (formData.description) fd.append('description', formData.description);

        if (editingId) {
          await specialtyService.update(editingId, fd);
          alert('Cập nhật chuyên khoa thành công');
        } else {
          await specialtyService.create(fd);
          alert('Thêm chuyên khoa thành công');
        }
      } else {
        // No file selected: send JSON payload. If image_url is empty string, remove it.
        const payload = { ...formData };
        if (payload.image_url !== undefined && String(payload.image_url).trim() === '') {
          delete payload.image_url;
        }

        if (editingId) {
          await specialtyService.update(editingId, payload);
          alert('Cập nhật chuyên khoa thành công');
        } else {
          await specialtyService.create(payload);
          alert('Thêm chuyên khoa thành công');
        }
      }
      handleCloseModal();
      loadSpecialties();
    } catch (e) {
      // Show useful message from API if available
      const msg = e?.response?.data?.message || e?.message || 'Lỗi server';
      alert('Lỗi: ' + msg);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc muốn xóa chuyên khoa này?')) return;

    try {
      await specialtyService.delete(id);
      alert('Xóa chuyên khoa thành công');
      loadSpecialties();
    } catch (e) {
      alert('Lỗi xóa: ' + (e?.message || 'Lỗi server'));
    }
  };

  const filtered = specialties.filter(s =>
    s.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    s.slug?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <AdminLayout>
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Quản lý chuyên khoa</h1>
          <button
            onClick={() => handleOpenModal()}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            <Plus size={20} /> Thêm chuyên khoa
          </button>
        </div>

        <div className="mb-4 flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-3 text-gray-400" size={18} />
            <input
              type="text"
              placeholder="Tìm kiếm chuyên khoa..."
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
            <div className="max-h-[60vh] overflow-y-auto">
              <table className="w-full">
              <thead className="bg-gray-50 border-b">
                  <tr>
                    <th className="text-left py-3 px-4 font-medium text-gray-700">ID</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-700">Ảnh</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-700">Tên chuyên khoa</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-700">Slug</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-700">Mô tả</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-700">Hành động</th>
                  </tr>
                </thead>
              <tbody>
                {filtered.map(specialty => (
                  <tr key={specialty.id} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm text-gray-600">#{specialty.id}</td>
                    <td className="py-3 px-4 text-sm font-medium text-gray-900">{specialty.name}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {specialty.image_url ? (
                        <img
                          src={specialty.image_url}
                          alt={specialty.name}
                          className="w-12 h-12 object-cover rounded cursor-pointer"
                          onClick={() => { setViewerSrc(specialty.image_url); setShowViewer(true); }}
                        />
                      ) : (
                        <div className="w-12 h-12 bg-gray-100 rounded flex items-center justify-center text-gray-400">No</div>
                      )}
                    </td>
                    <td className="py-3 px-4 text-sm font-medium text-gray-900">{specialty.name}</td>
                    <td className="py-3 px-4 text-sm text-gray-600">{specialty.slug || '-'}</td>
                    <td className="py-3 px-4 text-sm text-gray-600 max-w-xs truncate">{specialty.description || '-'}</td>
                    <td className="py-3 px-4 text-sm space-x-2">
                      <button
                        onClick={() => handleOpenModal(specialty)}
                        className="text-blue-600 hover:text-blue-800 inline-flex items-center gap-1"
                        title="Sửa"
                      >
                        <Edit2 size={16} />
                      </button>
                      <button
                        onClick={() => handleDelete(specialty.id)}
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
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">Không có chuyên khoa nào</div>
        )}

        {/* Modal */}
        {showModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full">
              <div className="flex justify-between items-center p-6 border-b">
                <h2 className="text-xl font-bold">{editingId ? 'Sửa chuyên khoa' : 'Thêm chuyên khoa'}</h2>
                <button onClick={handleCloseModal} className="text-gray-500 hover:text-gray-700">
                  <X size={24} />
                </button>
              </div>

              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Tên chuyên khoa *</label>
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
                    placeholder="Mô tả chuyên khoa"
                    rows="3"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Hình ảnh</label>
                  <div className="flex items-center gap-3">
                    <div className="w-20 h-20 bg-gray-100 rounded overflow-hidden cursor-pointer" onClick={() => fileRef.current && fileRef.current.click()}>
                      {imagePreview ? (
                        <img src={imagePreview} alt="specialty" className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-gray-400">No</div>
                      )}
                    </div>
                    <div>
                      <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => {
                        const f = e.target.files && e.target.files[0];
                        setImageFile(f || null);
                        if (f) {
                          try { if (imagePreview && imagePreview.startsWith && imagePreview.startsWith('blob:')) URL.revokeObjectURL(imagePreview); } catch (err) {}
                          setImagePreview(URL.createObjectURL(f));
                        }
                      }} />
                      <div className="flex gap-2">
                        <button type="button" onClick={() => fileRef.current && fileRef.current.click()} className="px-3 py-1 border rounded bg-white">Chọn ảnh</button>
                        <input
                          type="text"
                          value={formData.image_url}
                          onChange={(e) => setFormData({ ...formData, image_url: e.target.value })}
                          className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                          placeholder="Hoặc nhập URL hình ảnh..."
                        />
                      </div>
                    </div>
                  </div>
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
        {showViewer && (
          <ImageViewer src={viewerSrc} alt="Specialty image" onClose={() => setShowViewer(false)} />
        )}
      </div>
    </AdminLayout>
  );
}