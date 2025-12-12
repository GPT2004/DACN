import React, { useEffect, useState } from 'react';
import AdminLayout from '../../components/admin/AdminLayout';
import { auditLogService } from '../../services/auditLogService';

export default function AuditLogsPage() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);

  useEffect(() => {
    loadLogs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

  const loadLogs = async () => {
    try {
      setLoading(true);
      const res = await auditLogService.getAllAuditLogs({ page, limit: 50 });
      setLogs(res.logs || []);
    } catch (e) {
      console.error('Failed to load audit logs', e);
      setLogs([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AdminLayout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Audit Logs</h1>
            <p className="text-sm text-gray-600">Lịch sử thao tác hệ thống (Admin)</p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-600">#</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-600">Thời gian</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-600">Người</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-600">Hành động</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-600">Meta</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan="5" className="p-6 text-center">Đang tải...</td></tr>
                ) : logs.length === 0 ? (
                  <tr><td colSpan="5" className="p-6 text-center">Chưa có bản ghi</td></tr>
                ) : (
                  logs.map((l, i) => (
                    <tr key={l.id} className="border-b hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm text-gray-700">{(page-1)*50 + i + 1}</td>
                      <td className="px-4 py-3 text-sm text-gray-700">{new Date(l.created_at).toLocaleString()}</td>
                      <td className="px-4 py-3 text-sm text-gray-700">{l.user ? l.user.full_name || l.user.email : 'System'}</td>
                      <td className="px-4 py-3 text-sm text-gray-700">{l.action}</td>
                      <td className="px-4 py-3 text-sm text-gray-700"><pre className="whitespace-pre-wrap">{JSON.stringify(l.meta || {}, null, 2)}</pre></td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="flex justify-between items-center">
          <button onClick={() => setPage(p => Math.max(1, p-1))} className="px-4 py-2 bg-gray-200 rounded">Trước</button>
          <div>Trang {page}</div>
          <button onClick={() => setPage(p => p+1)} className="px-4 py-2 bg-gray-200 rounded">Sau</button>
        </div>
      </div>
    </AdminLayout>
  );
}
