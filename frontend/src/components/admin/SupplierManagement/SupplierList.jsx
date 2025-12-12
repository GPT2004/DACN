// frontend/src/components/admin/SupplierManagement/SupplierList.jsx
import React, { useState, useEffect } from 'react';
import { Search, Plus, Edit, Trash2, Eye, Phone, Mail } from 'lucide-react';
import { supplierService } from '../../../services/supplierService';
import { formatDate } from '../../../utils/helpers';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Badge from '../../common/Badge';
import Loader from '../../common/Loader';
import Pagination from '../../common/Pagination';

export default function SupplierList({ onEdit, onCreate, onView }) {
  const [suppliers, setSuppliers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  useEffect(() => {
    fetchSuppliers();
  }, [page, searchTerm]);

  const fetchSuppliers = async () => {
    try {
      setLoading(true);
      const response = await supplierService.getAllSuppliers({
        page,
        limit: 10,
        search: searchTerm,
      });
      setSuppliers(response.data.suppliers || []);
      setTotalPages(response.data.pagination?.totalPages || 1);
    } catch (error) {
      console.error('Fetch suppliers error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa nhà cung cấp này?')) return;
    
    try {
      await supplierService.deleteSupplier(id);
      fetchSuppliers();
    } catch (error) {
      console.error('Delete supplier error:', error);
      alert('Không thể xóa nhà cung cấp này!');
    }
  };

  if (loading) {
    return <Loader />;
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Quản lý Nhà cung cấp</h2>
        <Button onClick={onCreate}>
          <Plus className="w-4 h-4 mr-2" />
          Thêm Nhà cung cấp
        </Button>
      </div>

      {/* Search */}
      <div className="flex gap-4">
        <div className="flex-1">
          <Input
            placeholder="Tìm kiếm theo tên, email, số điện thoại..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Nhà cung cấp
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Liên hệ
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Địa chỉ
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
            {suppliers.map((supplier) => (
              <tr key={supplier.id} className="hover:bg-gray-50">
                <td className="px-6 py-4">
                  <div className="text-sm font-medium text-gray-900">
                    {supplier.name}
                  </div>
                  {supplier.code && (
                    <div className="text-sm text-gray-500">
                      Mã: {supplier.code}
                    </div>
                  )}
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center text-sm text-gray-900 mb-1">
                    <Phone className="w-4 h-4 mr-2 text-gray-400" />
                    {supplier.phone}
                  </div>
                  <div className="flex items-center text-sm text-gray-500">
                    <Mail className="w-4 h-4 mr-2 text-gray-400" />
                    {supplier.email}
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="text-sm text-gray-900">
                    {supplier.address || 'N/A'}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {formatDate(supplier.created_at)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <Badge 
                    text={supplier.is_active ? 'Hoạt động' : 'Ngừng hoạt động'}
                  />
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <div className="flex justify-end gap-2">
                    <button
                      onClick={() => onView(supplier)}
                      className="text-blue-600 hover:text-blue-900"
                      title="Xem chi tiết"
                    >
                      <Eye className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => onEdit(supplier)}
                      className="text-yellow-600 hover:text-yellow-900"
                      title="Chỉnh sửa"
                    >
                      <Edit className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => handleDelete(supplier.id)}
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

        {suppliers.length === 0 && (
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