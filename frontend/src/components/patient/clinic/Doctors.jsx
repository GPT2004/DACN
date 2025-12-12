import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { getAllDoctorsPublic } from '../../../services/doctorService';

export default function Doctors({ specialty = null, onSelectSpecialty = null }) {
  const navigate = useNavigate();
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        setLoading(true);
        const params = { limit: 6 };
        if (specialty) params.specialty = specialty;
        const res = await getAllDoctorsPublic(params);
        if (!mounted) return;
        setDoctors(res.data.doctors || res.data || []);
      } catch (error) {
        console.error('Load doctors error', error);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => { mounted = false; };
  }, [specialty]);

  if (loading) return <div>Đang tải bác sĩ...</div>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {doctors.length === 0 && (
        <div className="text-gray-500">Chưa có bác sĩ để hiển thị.</div>
      )}
        {doctors.map((d) => {
          const id = d.id || d.userId || (d.user && d.user.id) || d._id;
          const user = d.user || {};
          const avatar = d.avatar_url || user.avatar_url || user.avatar || d.photo || d.profilePicture;
          const name = user.full_name || d.fullName || d.name || user.name || '';
          const specialtiesText = d.specialty || (d.specialties && d.specialties.join(', ')) || user.specialties?.join(', ') || '';

          const initials = name ? name.split(' ').map(n => n[0]).slice(0,2).join('') : 'BS';

          // try extract experience from bio
          const experience = (() => {
            const bio = d.bio || d.description || user.bio || '';
            const m = String(bio || '').match(/(\d{1,2})\+?\s*năm/iu);
            return m ? `${m[1]} năm` : null;
          })();

          return (
            <div key={id || Math.random()} className="border rounded-lg p-4 flex items-start gap-4">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center overflow-hidden">
                {avatar ? (
                  <img src={avatar} alt={name} className="w-full h-full object-cover" />
                ) : (
                  <span className="text-xl text-gray-700">{initials}</span>
                )}
              </div>

              <div className="flex-1">
                <div className="font-semibold text-lg">{name || 'Bác sĩ'}</div>
                <div className="text-sm text-gray-600">{specialtiesText}</div>
                {experience && (
                  <div className="text-sm text-gray-500">Kinh nghiệm: {experience}</div>
                )}

                <div className="mt-3 flex gap-2">
                  <Link
                    to={`/doctors/${id}`}
                    className="px-3 py-1 bg-emerald-600 text-white rounded text-sm"
                  >
                    Xem chi tiết
                  </Link>
                </div>
              </div>
            </div>
          );
        })}
    </div>
  );
}
