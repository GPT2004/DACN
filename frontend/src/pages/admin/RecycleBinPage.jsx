import React, { useEffect, useState } from 'react';
import AdminLayout from '../../components/admin/AdminLayout';
import { medicineService } from '../../services/medicineService';
import { userService } from '../../services/userService';

export default function RecycleBinPage() {
  const [medicines, setMedicines] = useState([]);
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDeleted();
  }, []);

  const loadDeleted = async () => {
    try {
      setLoading(true);
      const m = await medicineService.getAll({ include_deleted: true });
      const u = await userService.getAll({ include_deleted: true });
      setMedicines(m || []);
      setUsers(u || []);
    } catch (e) {
      console.error('Failed to load recycle bin', e);
      setMedicines([]);
      setUsers([]);
    } finally {
      setLoading(false);
    }
  };

  const handleRestoreMedicine = async (id) => {
    await medicineService.restore(id);
    loadDeleted();
  };

  const handleRestoreUser = async (id) => {
    await userService.restore(id);
    loadDeleted();
  };

  return (
    <AdminLayout>
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Thùng rác</h1>

        {loading ? (
          <div>Đang tải...</div>
        ) : (
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white rounded shadow p-4">
              <h2 className="font-semibold mb-3">Thuốc đã xóa</h2>
              {medicines.length === 0 ? (
                <div className="text-sm text-gray-500">Không có thuốc bị xóa</div>
              ) : (
                <ul className="space-y-2">
                  {medicines.map(m => (
                    <li key={m.id} className="flex items-center justify-between border-b py-2">
                      <div>
                        <div className="font-medium">{m.name}</div>
                        <div className="text-xs text-gray-500">SKU: {m.sku || '-'}</div>
                      </div>
                      <div className="space-x-2">
                        <button onClick={() => handleRestoreMedicine(m.id)} className="px-3 py-1 bg-green-500 text-white rounded">Restore</button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="bg-white rounded shadow p-4">
              <h2 className="font-semibold mb-3">Người dùng đã xóa</h2>
              {users.length === 0 ? (
                <div className="text-sm text-gray-500">Không có user bị xóa</div>
              ) : (
                <ul className="space-y-2">
                  {users.map(u => (
                    <li key={u.id} className="flex items-center justify-between border-b py-2">
                      <div>
                        <div className="font-medium">{u.full_name || u.email}</div>
                        <div className="text-xs text-gray-500">Role: {u.role?.name || u.role}</div>
                      </div>
                      <div className="space-x-2">
                        <button onClick={() => handleRestoreUser(u.id)} className="px-3 py-1 bg-green-500 text-white rounded">Restore</button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </div>
    </AdminLayout>
  );
}
