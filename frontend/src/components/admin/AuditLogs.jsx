// frontend/src/components/admin/AuditLogs.jsx
import React, { useState, useEffect } from 'react';
import { Search, Filter, Download, Eye } from 'lucide-react';
import { auditLogService } from '../../services/auditLogService';
import { formatDateTime } from '../../utils/helpers';
import Button from '../common/Button';
import Input from '../common/Input';
import Badge from '../common/Badge';
import Loader from '../common/Loader';
import Pagination from '../common/Pagination';
import Modal from '../common/Modal';

export default function AuditLogs() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [actionFilter, setActionFilter] = useState('');
  const [actionTypes, setActionTypes] = useState([]);
  const [selectedLog, setSelectedLog] = useState(null);
  const [showDetail, setShowDetail] = useState(false);

  useEffect(() => {
    fetchActionTypes();
  }, []);

  useEffect(() => {
    fetchLogs();
  }, [page, searchTerm, actionFilter]);

  const fetchActionTypes = async () => {
    try {
      const response = await auditLogService.getActionTypes();
      setActionTypes(response.data.types || []);
    } catch (error) {
      console.error('Fetch action types error:', error);
    }
  };

  const fetchLogs = async () => {
    try {
      setLoading(true);
      const response = await auditLogService.getAllAuditLogs({
        page,
        limit: 20,
        search: searchTerm,
        action: actionFilter
      });
      setLogs(response.data.logs || []);
      setTotalPages(response.data.pagination?.totalPages || 1);
    } catch (error) {
      console.error('Fetch logs error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetail = (log) => {
    setSelectedLog(log);
    setShowDetail(true);
  };

  const handleExport = async () => {
    try {
      // Implementation for exporting logs
      alert('Chức năng xuất báo cáo đang được phát triển');
    } catch (error) {
      console.error('Export error:', error);
    }
  };

  const getActionBadge = (action) => {
    const actionColors = {
      CREATE: 'bg-green-100 text-green-800',
      UPDATE: 'bg-blue-100 text-blue-800',
      DELETE: 'bg-red-100 text-red-800',
      LOGIN: 'bg-purple-100 text-purple-800',
      LOGOUT: 'bg-gray-100 text-gray-800',
      VIEW: 'bg-yellow-100 text-yellow-800',
    };

    const color = actionColors[action] || 'bg-gray-100 text-gray-800';
    return <span className={`px-2 py-1 text-xs font-medium rounded-full ${color}`}>{action}</span>;
  };

  if (loading) {
    return <Loader />;
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">Nhật ký Hệ thống</h2>
        <Button onClick={handleExport}>
          <Download className="w-4 h-4 mr-2" />
          Xuất báo cáo
        </Button>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2">
            <Input
              placeholder="Tìm kiếm theo người dùng, hành động..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div>
            <select
              className="w-full px-4 py-2 border border-gray-300 rounded-lg"
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value)}
            >
              <option value="">Tất cả hành động</option>
              {actionTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Logs Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Thời gian
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Người dùng
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Hành động
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Mô tả
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                IP Address
              </th>
              <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                Chi tiết
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {logs.map((log) => (
              <tr key={log.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {formatDateTime(log.created_at)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">
                    {log.user?.full_name || 'N/A'}
                  </div>
                  <div className="text-sm text-gray-500">
                    {log.user?.email || ''}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {getActionBadge(log.action)}
                </td>
                <td className="px-6 py-4">
                  <div className="text-sm text-gray-900">
                    {log.description || 'N/A'}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {log.ip_address || 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-center">
                  <button
                    onClick={() => handleViewDetail(log)}
                    className="text-blue-600 hover:text-blue-900"
                    title="Xem chi tiết"
                  >
                    <Eye className="w-5 h-5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {logs.length === 0 && (
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

      {/* Detail Modal */}
      {showDetail && selectedLog && (
        <Modal isOpen={showDetail} onClose={() => setShowDetail(false)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full">
            <h3 className="text-lg font-semibold mb-4">Chi tiết Nhật ký</h3>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-gray-600">Thời gian:</label>
                <div className="text-sm text-gray-900 mt-1">
                  {formatDateTime(selectedLog.created_at)}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-600">Người dùng:</label>
                <div className="text-sm text-gray-900 mt-1">
                  {selectedLog.user?.full_name} ({selectedLog.user?.email})
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-600">Hành động:</label>
                <div className="mt-1">
                  {getActionBadge(selectedLog.action)}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-600">Mô tả:</label>
                <div className="text-sm text-gray-900 mt-1">
                  {selectedLog.description || 'N/A'}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-600">IP Address:</label>
                <div className="text-sm text-gray-900 mt-1">
                  {selectedLog.ip_address || 'N/A'}
                </div>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-600">User Agent:</label>
                <div className="text-sm text-gray-900 mt-1 break-all">
                  {selectedLog.user_agent || 'N/A'}
                </div>
              </div>

              {selectedLog.meta && (
                <div>
                  <label className="text-sm font-medium text-gray-600">Metadata:</label>
                  <pre className="text-xs text-gray-900 mt-1 bg-gray-50 p-3 rounded overflow-auto max-h-40">
                    {JSON.stringify(selectedLog.meta, null, 2)}
                  </pre>
                </div>
              )}
            </div>

            <div className="mt-6 flex justify-end">
              <Button onClick={() => setShowDetail(false)}>Đóng</Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}