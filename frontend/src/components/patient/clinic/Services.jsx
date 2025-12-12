import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getSpecialties } from '../../../services/doctorService';

export default function Services({ onSelectSpecialty }) {
  const [specialties, setSpecialties] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        setLoading(true);
        const res = await getSpecialties();
        const list = Array.isArray(res.data) ? res.data : (res.data?.specialties || []);
        if (!mounted) return;
        setSpecialties(list);
      } catch (err) {
        console.error('Load specialties error', err);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => { mounted = false; };
  }, []);

  return (
    <section id="services">
      <h2 className="text-2xl font-bold mb-4">Dịch vụ & Chuyên khoa</h2>
      {loading && <div>Đang tải chuyên khoa...</div>}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {specialties.length === 0 && !loading && (
          <div className="text-gray-500">Không có chuyên khoa để hiển thị.</div>
        )}
        {specialties.map((s) => {
          // allow backend to return either a string or an object
          const name = typeof s === 'string' ? s : s.name || s.title || s.specialty;
          const id = typeof s === 'string' ? s : s.id || name;
          const image = s && typeof s === 'object' ? (s.image || s.image_url || s.icon) : null;

          return (
            <Link
              key={id}
              to={`/specialty/${encodeURIComponent(name)}`}
              onClick={() => onSelectSpecialty?.(name)}
              className="flex items-center gap-4 text-left border rounded-lg p-4 hover:shadow-md hover:border-emerald-300 transition"
            >
              <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center overflow-hidden">
                {image ? (
                  <img src={image} alt={name} className="w-full h-full object-cover" />
                ) : (
                  <span className="text-2xl text-gray-600">{String(name).charAt(0)}</span>
                )}
              </div>

              <div>
                <div className="font-semibold mb-1">{name}</div>
                {s && typeof s === 'object' && s.description ? (
                  <div className="text-sm text-gray-600">{s.description}</div>
                ) : (
                  <div className="text-sm text-gray-600">Xem bác sĩ và lịch khám thuộc chuyên khoa này →</div>
                )}
              </div>
            </Link>
          );
        })}
      </div>
    </section>
  );
}
