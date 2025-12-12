// frontend/src/components/admin/MedicineManagement/MedicineList.jsx
import React, { useState, useEffect } from 'react';
import { Search, Plus, Edit, Trash2, Eye, AlertTriangle } from 'lucide-react';
import { medicineService } from '../../../services/medicineService';
import { formatCurrency, formatNumber } from '../../../utils/helpers';
import Button from '../../common/Button';
import Input from '../../common/Input';
import Badge from '../../common/Badge';
import Loader from '../../common/Loader';
import Pagination from '../../common/Pagination';

export default function MedicineList({ onEdit, onCreate }) {
  const [medicines, setMedicines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [category, setCategory] = useState('');

  useEffect(() => {
    fetchMedicines();
  }, [page, searchTerm, category]);

  const fetchMedicines = async () => {
    try {
      setLoading(true);
      const response = await medicineService.getMedicines({
        page,
        limit: 10,
        search: searchTerm,
        category
      });
      setMedicines(response.data.medicines || []);
      setTotalPages(response.data.pagination?.totalPages || 1);
    } catch (error) {
      console.error('Fetch medicines error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa thuốc này?')) return;
    
    try {
      await medicineService.deleteMedicine(id);
      fetchMedicines();
    } catch (error) {
      console.error('Delete medicine error:', error);
      alert('Không thể xóa thuốc này!');
    }
  };

  const getStockStatus = (stock) => {
    if (stock === 0) return { text: 'Hết hàng', color: 'text-red-600 bg-red-50' };
    if (stock < 20) return { text: 'Sắp hết', color: 'text-yellow-600 bg-yellow-50' };
    return { text: 'Còn hàng', color: 'text-green-600 bg-green-50' };
  };

  if (loading) {
    return <Loader />;
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Quản lý Thuốc</h2>
        <Button onClick={onCreate}>
          <Plus className="w-4 h-4 mr-2" />
          Thêm Thuốc
        </Button>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="flex-1">
          <Input
            placeholder="Tìm kiếm theo tên, mã thuốc..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <select
          className="px-4 py-2 border border-gray-300 rounded-lg"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
        >
          <option value="">Tất cả loại</option>
          <option value="Kháng sinh">Kháng sinh</option>
          <option value="Giảm đau">Giảm đau</option>
          <option value="Hạ sốt">Hạ sốt</option>
          <option value="Vitamin">Vitamin</option>
          <option value="Tiêu hóa">Tiêu hóa</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Mã thuốc
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Tên thuốc
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Loại
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Đơn vị
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Giá bán
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Tồn kho
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
            {medicines.map((medicine) => {
              const stockStatus = getStockStatus(medicine.total_stock || 0);
              return (
                <tr key={medicine.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {medicine.code}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm font-medium text-gray-900">
                      {medicine.name}
                    </div>
                    {medicine.active_ingredient && (
                      <div className="text-sm text-gray-500">
                        {medicine.active_ingredient}
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Badge text={medicine.category || 'Khác'} />
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {medicine.unit}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {formatCurrency(medicine.price)}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="text-sm font-medium text-gray-900">
                        {formatNumber(medicine.total_stock || 0)}
                      </span>
                      {(medicine.total_stock || 0) < 20 && (
                        <AlertTriangle className="w-4 h-4 ml-2 text-yellow-500" />
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${stockStatus.color}`}>
                      {stockStatus.text}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex justify-end gap-2">
                      <button
                        onClick={() => onEdit(medicine)}
                        className="text-yellow-600 hover:text-yellow-900"
                        title="Chỉnh sửa"
                      >
                        <Edit className="w-5 h-5" />
                      </button>
                      <button
                        onClick={() => handleDelete(medicine.id)}
                        className="text-red-600 hover:text-red-900"
                        title="Xóa"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {medicines.length === 0 && (
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