import React, { useEffect, useState } from "react";

export default function SuppliersPage() {
  const [suppliers, setSuppliers] = useState([]);

  useEffect(() => {
    // TODO: fetch suppliers tại đây
    setSuppliers([]);
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Nhà cung cấp</h1>
      <div className="bg-white p-4 rounded shadow">
        {suppliers.length ? (
          suppliers.map((s) => <div key={s.id}>{s.name}</div>)
        ) : (
          "Không có nhà cung cấp"
        )}
      </div>
    </div>
  );
}
