import React, { useState, useEffect } from 'react';
import PharmacistLayout from '../../components/layout/PharmacistLayout';
import { getAllMedicines, importMedicines } from '../../services/medicineService';
import { ensureArray } from '../../utils/normalize';
import { Trash2, Edit2, Plus, Pill, Upload, Download, AlertTriangle, HelpCircle } from 'lucide-react';
import EditMedicineModal from '../../components/admin/EditMedicineModal';
import DeleteMedicineModal from '../../components/admin/DeleteMedicineModal';
import AddStockModal from '../../components/admin/AddStockModal';
import CreateMedicineModal from '../../components/admin/CreateMedicineModal';
import MedicineDetailModal from '../../components/admin/MedicineDetailModal';
import { importFromExcel, parseMedicinesFromExcel, exportToExcel, formatMedicinesForExport } from '../../utils/exportHelpers';

export default function MedicinesPage() {
  const [meds, setMeds] = useState([]);
  const [name, setName] = useState(''); // search by name only
  const [formError, setFormError] = useState('');
  const [loading, setLoading] = useState(true);
  const [editingMedicine, setEditingMedicine] = useState(null);
  const [deletingMedicine, setDeletingMedicine] = useState(null);
  const [addingStockFor, setAddingStockFor] = useState(null);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [detailMedicine, setDetailMedicine] = useState(null);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [importing, setImporting] = useState(false);
  const [importResults, setImportResults] = useState(null);
  const [showImportHelp, setShowImportHelp] = useState(false);

  useEffect(() => {
    loadMedicines();
  }, []);

  // loadMedicines accepts optional search params
  const loadMedicines = async (params = {}) => {
    try {
      setLoading(true);
      const r = await getAllMedicines(params);
      // Normalize array from various response shapes
      const arr = ensureArray(r);
      const unwrapped = arr && arr.length ? arr
        : r?.data?.data?.medicines
        || r?.data?.medicines
        || r?.medicines
        || [];
      setMeds(unwrapped);
    } catch (e) {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  // search action - real-time search by name only
  const search = async (searchName) => {
    setFormError('');
    try {
      await loadMedicines({ search: searchName.trim() });
    } catch (e) {
      setFormError('Tìm thuốc thất bại');
    }
  };

  const handleNameChange = (e) => {
    const value = e.target.value;
    setName(value);
    // Real-time search as user types
    search(value);
  };

  // Handle Excel import
  const handleImportExcel = async (file) => {
    try {
      setImporting(true);
      const excelData = await importFromExcel(file);
      const { medicines, errors } = parseMedicinesFromExcel(excelData);
      
      if (errors.length > 0) {
        alert(`Có lỗi trong file Excel:\n${errors.join('\n')}`);
        return;
      }
      
      if (medicines.length === 0) {
        alert('Không có dữ liệu thuốc hợp lệ trong file Excel');
        return;
      }
      
      const response = await importMedicines(medicines);
      setImportResults(response.data);
      
      // Reload medicines list
      await loadMedicines();
      
      const { success = [], errors: importErrors = [] } = response.data || {};
      if (importErrors.length > 0) {
        console.error('Import medicines errors:', importErrors);
        const sampleErrors = importErrors.slice(0, 5).map((e, idx) => `${idx + 1}) ${e.error || JSON.stringify(e)}`).join('\n');
        alert(`Import thành công: ${success.length} thuốc\nLỗi: ${importErrors.length} thuốc\nChi tiết:\n${sampleErrors}`);
      } else {
        alert(`Import thành công: ${success.length} thuốc`);
      }
      
    } catch (error) {
      console.error('Import error:', error);
      alert('Lỗi import: ' + (error.response?.data?.message || error.message));
    } finally {
      setImporting(false);
    }
  };

  // Handle export to Excel
  const handleExportExcel = () => {
    const exportData = formatMedicinesForExport(meds);
    exportToExcel(exportData, 'danh-sach-thuoc.xlsx', 'Thuoc');
  };

  return (
    <PharmacistLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Pill className="text-orange-600" size={32} />
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Quản lý thuốc</h1>
                <p className="text-gray-600 mt-1">Quản lý kho thuốc và thông tin chi tiết</p>
              </div>
            </div>
          </div>
        </div>

        {/* Search Form - Tìm theo tên thuốc (real-time) */}
        <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Tìm thuốc theo tên</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
            <input
              className="col-span-1 md:col-span-2 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
              value={name}
              onChange={handleNameChange}
              placeholder="Nhập tên thuốc (tìm kiếm tự động)..."
            />
            <div className="flex items-center gap-3 col-span-1 md:col-span-3">
              <button
                onClick={() => setCreateModalOpen(true)}
                className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium flex items-center gap-2 justify-center"
              >
                <Plus size={16} />
                Thêm thuốc
              </button>
              <button
                onClick={() => document.getElementById('excel-import').click()}
                disabled={importing}
                className="px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition font-medium flex items-center gap-2 disabled:opacity-50"
              >
                <Upload size={16} />
                {importing ? 'Đang import...' : 'Import Excel'}
              </button>
              <button
                onClick={() => setShowImportHelp(true)}
                className="px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition font-medium flex items-center gap-2"
                title="Hướng dẫn format Excel"
              >
                <HelpCircle size={16} />
                Hướng dẫn
              </button>
              <button
                onClick={handleExportExcel}
                className="px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition font-medium flex items-center gap-2"
              >
                <Download size={16} />
                Export Excel
              </button>
            </div>
            <input
              id="excel-import"
              type="file"
              accept=".xlsx,.xls"
              onChange={(e) => {
                const file = e.target.files[0];
                if (file) {
                  handleImportExcel(file);
                }
                e.target.value = ''; // Reset input
              }}
              className="hidden"
            />
          </div>
          {formError && <div className="text-sm text-red-600 mt-2">{formError}</div>}
        </div>

        {/* Table */}
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700 w-12">#</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Tên thuốc</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Số lượng trong kho</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Hạn sử dụng gần nhất</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700">Trạng thái</th>
                  <th className="px-6 py-4 text-center text-sm font-semibold text-gray-700 w-24">Hành động</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan="6" className="px-6 py-12 text-center">
                      <div className="flex justify-center">
                        <div className="spinner"></div>
                      </div>
                    </td>
                  </tr>
                ) : meds.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="px-6 py-12 text-center text-gray-500">
                      Chưa có thuốc nào trong hệ thống
                    </td>
                  </tr>
                ) : (
                  meds.map((m, i) => {
                    const stock = m.total_stock || 0;
                    const status = stock === 0 ? 'Hết hàng' : stock < 10 ? 'Sắp hết' : 'Có sẵn';
                    const statusColor = stock === 0 ? 'bg-red-100 text-red-800' : stock < 10 ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800';
                    
                    // Find nearest expiry date
                    const nearestExpiry = m.stocks && m.stocks.length > 0 
                      ? m.stocks
                          .filter(s => new Date(s.expiry_date) > new Date())
                          .sort((a, b) => new Date(a.expiry_date) - new Date(b.expiry_date))[0]
                      : null;
                    
                    const expiryDate = nearestExpiry ? new Date(nearestExpiry.expiry_date) : null;
                    const daysUntilExpiry = expiryDate ? Math.ceil((expiryDate - new Date()) / (1000 * 60 * 60 * 24)) : null;
                    
                    let expiryStatus = 'Không có thông tin';
                    let expiryColor = 'bg-gray-100 text-gray-800';
                    
                    if (daysUntilExpiry !== null) {
                      if (daysUntilExpiry < 0) {
                        expiryStatus = 'Đã hết hạn';
                        expiryColor = 'bg-red-100 text-red-800';
                      } else if (daysUntilExpiry <= 30) {
                        expiryStatus = `Hết hạn trong ${daysUntilExpiry} ngày`;
                        expiryColor = 'bg-orange-100 text-orange-800';
                      } else {
                        expiryStatus = expiryDate.toLocaleDateString('vi-VN');
                        expiryColor = 'bg-green-100 text-green-800';
                      }
                    }

                    return (
                      <tr key={m.id || i} className="border-b border-gray-200 hover:bg-gray-50 transition cursor-pointer" onClick={() => setDetailMedicine(m)}>
                        <td className="px-6 py-4 text-sm text-gray-600">{i + 1}</td>
                        <td className="px-6 py-4 text-sm font-medium text-gray-900">{m.name}</td>
                        <td className="px-6 py-4 text-sm text-gray-600">
                          <span className="font-semibold">{stock}</span> {m.unit || 'cái'}
                        </td>
                        <td className="px-6 py-4 text-sm">
                          <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${expiryColor}`}>
                            {expiryStatus}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${statusColor}`}>
                            {status}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center justify-center gap-2">
                              <button
                                onClick={() => setEditingMedicine(m)}
                                className="p-2 text-blue-600 hover:bg-blue-100 rounded-lg transition"
                                title="Chỉnh sửa & nhập kho"
                              >
                              <Edit2 size={18} />
                            </button>
                              <button
                                onClick={() => setDeletingMedicine(m)}
                                className="p-2 text-red-600 hover:bg-red-100 rounded-lg transition"
                                title="Xóa"
                              >
                              <Trash2 size={18} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer Info */}
        {meds.length > 0 && (
          <div className="text-sm text-gray-600">
            Tổng số thuốc: <span className="font-semibold">{meds.length}</span> loại
          </div>
        )}
      </div>
      
        {/* Modals */}
        {editingMedicine && (
          <EditMedicineModal
            medicine={editingMedicine}
            onClose={() => setEditingMedicine(null)}
            onSuccess={() => { setEditingMedicine(null); loadMedicines(); }}
          />
        )}

        {deletingMedicine && (
          <DeleteMedicineModal
            medicine={deletingMedicine}
            onClose={() => setDeletingMedicine(null)}
            onSuccess={() => { setDeletingMedicine(null); loadMedicines(); }}
          />
        )}
        {addingStockFor && (
          <AddStockModal
            medicine={addingStockFor}
            onClose={() => setAddingStockFor(null)}
            onSuccess={() => { setAddingStockFor(null); loadMedicines(); }}
          />
        )}
        {createModalOpen && (
          <CreateMedicineModal
            onClose={() => setCreateModalOpen(false)}
            onSuccess={() => { setCreateModalOpen(false); loadMedicines(); }}
          />
        )}

        {detailMedicine && (
          <MedicineDetailModal
            medicine={detailMedicine}
            onClose={() => setDetailMedicine(null)}
          />
        )}

        {/* Import Help Modal */}
        {showImportHelp && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Hướng dẫn Import Excel</h3>
                <button
                  onClick={() => setShowImportHelp(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2">Format file Excel yêu cầu:</h4>
                  <p className="text-sm text-gray-600 mb-3">
                    File Excel phải có sheet đầu tiên chứa dữ liệu với các cột sau (theo thứ tự):
                  </p>
                  
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2 font-semibold">Cột</th>
                          <th className="text-left py-2 font-semibold">Tên cột (hàng đầu tiên)</th>
                          <th className="text-left py-2 font-semibold">Mô tả</th>
                          <th className="text-left py-2 font-semibold">Bắt buộc</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b">
                          <td className="py-2">A</td>
                          <td className="py-2 font-mono">name</td>
                          <td className="py-2">Tên thuốc</td>
                          <td className="py-2 text-green-600">✓</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-2">B</td>
                          <td className="py-2 font-mono">code</td>
                          <td className="py-2">Mã thuốc (tùy chọn)</td>
                          <td className="py-2 text-gray-500">✗</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-2">C</td>
                          <td className="py-2 font-mono">quantity</td>
                          <td className="py-2">Số lượng nhập kho</td>
                          <td className="py-2 text-green-600">✓</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-2">D</td>
                          <td className="py-2 font-mono">expiry_date</td>
                          <td className="py-2">Hạn sử dụng (DD/MM/YYYY)</td>
                          <td className="py-2 text-green-600">✓</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-2">E</td>
                          <td className="py-2 font-mono">batch_number</td>
                          <td className="py-2">Số lô (tùy chọn)</td>
                          <td className="py-2 text-gray-500">✗</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-2">F</td>
                          <td className="py-2 font-mono">unit</td>
                          <td className="py-2">Đơn vị (ví dụ: viên, hộp)</td>
                          <td className="py-2 text-gray-500">✗</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-2">G</td>
                          <td className="py-2 font-mono">price</td>
                          <td className="py-2">Giá tiền (VNĐ)</td>
                          <td className="py-2 text-gray-500">✗</td>
                        </tr>
                        <tr>
                          <td className="py-2">H</td>
                          <td className="py-2 font-mono">description</td>
                          <td className="py-2">Mô tả thuốc</td>
                          <td className="py-2 text-gray-500">✗</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold text-gray-800 mb-2">Lưu ý quan trọng:</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Hàng đầu tiên là tiêu đề cột, dữ liệu bắt đầu từ hàng thứ 2</li>
                    <li>• Cột <code>name</code>, <code>quantity</code>, <code>expiry_date</code> là bắt buộc</li>
                    <li>• Ngày tháng phải theo format DD/MM/YYYY (ví dụ: 31/12/2024)</li>
                    <li>• Số lượng phải là số dương</li>
                    <li>• Nếu thuốc đã tồn tại (theo tên hoặc mã), hệ thống sẽ thêm lô mới</li>
                    <li>• Nếu thuốc chưa tồn tại, hệ thống sẽ tạo thuốc mới</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold text-gray-800 mb-2">Ví dụ file Excel:</h4>
                  <div className="bg-gray-50 p-4 rounded-lg font-mono text-sm">
                    <div>name | code | quantity | expiry_date | batch_number | unit | price | description</div>
                    <div>Paracetamol | PARA001 | 100 | 31/12/2024 | LOT2024001 | viên | 500 | Thuốc giảm đau</div>
                    <div>Vitamin C | VIT001 | 50 | 15/06/2025 | LOT2024002 | hộp | 25000 | Vitamin bổ sung</div>
                  </div>
                </div>

                <div className="flex justify-end gap-3 pt-4 border-t">
                  <button
                    onClick={() => setShowImportHelp(false)}
                    className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition"
                  >
                    Đóng
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
    </PharmacistLayout>
  );
}
