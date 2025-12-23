/* eslint-disable no-console */
/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable no-unused-vars */
// frontend/src/components/admin/DoctorManagement/DoctorList.jsx
import React, { useState, useEffect } from 'react';
import { Search, Plus, Edit, Trash2, Eye, ToggleLeft, ToggleRight } from 'lucide-react';
import { doctorService } from '../../../services/doctorService';
import { formatDate } from '../../../utils/helpers';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Badge from '../../common/Badge';
import Loader from '../../common/Loader';
import Pagination from '../../common/Pagination';

export default function DoctorList({ onEdit, onView, onCreate }) {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [specialty, setSpecialty] = useState('');

  useEffect(() => {
    fetchDoctors();
  }, [page, searchTerm, specialty]);

  const fetchDoctors = async () => {
    try {
      setLoading(true);
      const response = await doctorService.getAllDoctors({
        page,
        limit: 10,
        search: searchTerm,
        specialty
      });
      setDoctors(response.data.doctors || []);
      setTotalPages(response.data.pagination?.totalPages || 1);
    } catch (error) {
      console.error('Fetch doctors error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleStatus = async (id) => {
    try {
      await doctorService.toggleDoctorStatus(id);
      fetchDoctors();
    } catch (error) {
      console.error('Toggle status error:', error);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa bác sĩ này?')) return;
    
    try {
      await doctorService.deleteDoctor(id);
      fetchDoctors();
    } catch (error) {
      console.error('Delete doctor error:', error);
    }
  };

  if (loading) {
    return <Loader />;
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Quản lý Bác sĩ</h2>
        <Button onClick={onCreate}>
          <Plus className="w-4 h-4 mr-2" />
          Thêm Bác sĩ
        </Button>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="flex-1">
          <Input
            placeholder="Tìm kiếm theo tên, email, chuyên khoa..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <select
          className="px-4 py-2 border border-gray-300 rounded-lg"
          value={specialty}
          onChange={(e) => setSpecialty(e.target.value)}
        >
          <option value="">Tất cả chuyên khoa</option>
          <option value="Nội khoa">Nội khoa</option>
          <option value="Ngoại khoa">Ngoại khoa</option>
          <option value="Sản phụ khoa">Sản phụ khoa</option>
          <option value="Nhi khoa">Nhi khoa</option>
          <option value="Tim mạch">Tim mạch</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Bác sĩ
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Chuyên khoa
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Liên hệ
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Trạng thái
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Ngày tạo
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                Thao tác
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {doctors.map((doctor) => (
              <tr key={doctor.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <img
                      className="h-10 w-10 rounded-full"
                      src={doctor.user?.avatar_url || '/avatar-placeholder.png'}
                      alt={doctor.user?.full_name}
                    />
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900">
                        {doctor.user?.full_name}
                      </div>
                      <div className="text-sm text-gray-500">
                        {doctor.user?.email}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <Badge text={doctor.specialty} />
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {doctor.user?.phone}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <Badge 
                    text={doctor.user?.is_active ? 'Hoạt động' : 'Ngừng hoạt động'}
                  />
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {formatDate(doctor.created_at)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => onView(doctor)}
                      className="text-blue-600 hover:text-blue-900"
                      title="Xem chi tiết"
                    >
                      <Eye className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => onEdit(doctor)}
                      className="text-yellow-600 hover:text-yellow-900"
                      title="Chỉnh sửa"
                    >
                      <Edit className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => handleToggleStatus(doctor.id)}
                      className="text-green-600 hover:text-green-900"
                      title="Đổi trạng thái"
                    >
                      {doctor.user?.is_active ? (
                        <ToggleRight className="w-5 h-5" />
                      ) : (
                        <ToggleLeft className="w-5 h-5" />
                      )}
                    </button>
                    <button
                      onClick={() => handleDelete(doctor.id)}
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

        {doctors.length === 0 && (
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