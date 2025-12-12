// frontend/src/components/admin/PatientManagement/PatientList.jsx
import React, { useState, useEffect } from 'react';
import { Search, Eye, Edit, Trash2, UserPlus } from 'lucide-react';
import { patientService } from '../../../services/patientService';
import { formatDate, calculateAge } from '../../../utils/helpers';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Badge from '../../common/Badge';
import Loader from '../../common/Loader';
import Pagination from '../../common/Pagination';

export default function PatientList({ onView, onEdit }) {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [gender, setGender] = useState('');

  useEffect(() => {
    fetchPatients();
  }, [page, searchTerm, gender]);

  const fetchPatients = async () => {
    try {
      setLoading(true);
      const response = await patientService.getAllPatients({
        page,
        limit: 10,
        search: searchTerm,
        gender
      });
      setPatients(response.data.patients || []);
      setTotalPages(response.data.pagination?.totalPages || 1);
    } catch (error) {
      console.error('Fetch patients error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa bệnh nhân này?')) return;
    
    try {
      await patientService.deletePatient(id);
      fetchPatients();
    } catch (error) {
      console.error('Delete patient error:', error);
      alert('Không thể xóa bệnh nhân này!');
    }
  };

  if (loading) {
    return <Loader />;
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Quản lý Bệnh nhân</h2>
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
          value={gender}
          onChange={(e) => setGender(e.target.value)}
        >
          <option value="">Tất cả giới tính</option>
          <option value="Male">Nam</option>
          <option value="Female">Nữ</option>
          <option value="Other">Khác</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Bệnh nhân
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Giới tính / Tuổi
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Liên hệ
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Nhóm máu
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Ngày đăng ký
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
            {patients.map((patient) => (
              <tr key={patient.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <img
                      className="h-10 w-10 rounded-full"
                      src={patient.user?.avatar_url || '/avatar-placeholder.png'}
                      alt={patient.user?.full_name}
                    />
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900">
                        {patient.user?.full_name}
                      </div>
                      <div className="text-sm text-gray-500">
                        Mã BN: {patient.id}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">
                    {patient.gender === 'Male' ? 'Nam' : patient.gender === 'Female' ? 'Nữ' : 'Khác'}
                  </div>
                  <div className="text-sm text-gray-500">
                    {calculateAge(patient.user?.dob)} tuổi
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{patient.user?.phone}</div>
                  <div className="text-sm text-gray-500">{patient.user?.email}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {patient.blood_type && (
                    <Badge text={patient.blood_type} />
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {formatDate(patient.created_at)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <Badge 
                    text={patient.user?.is_active ? 'Hoạt động' : 'Ngừng hoạt động'}
                  />
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => onView(patient)}
                      className="text-blue-600 hover:text-blue-900"
                      title="Xem chi tiết"
                    >
                      <Eye className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => handleDelete(patient.id)}
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

        {patients.length === 0 && (
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