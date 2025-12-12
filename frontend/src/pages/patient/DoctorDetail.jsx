import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getDoctorByIdPublic, getDoctorSchedules } from '../../services/doctorService';
import { useAuth } from '../../context/AuthContext';
import { Star } from 'lucide-react';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export default function DoctorDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();

  const [doctor, setDoctor] = useState(null);
  const [schedules, setSchedules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [reviews, setReviews] = useState([]);
  const [reviewStats, setReviewStats] = useState(null);
  const [userReview, setUserReview] = useState(null);
  const [showRatingModal, setShowRatingModal] = useState(false);
  const [ratingValue, setRatingValue] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [reviewComment, setReviewComment] = useState('');
  const [submittingReview, setSubmittingReview] = useState(false);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        setLoading(true);
        // debug: log the id we're loading
        // eslint-disable-next-line no-console
        console.log('DoctorDetail: loading doctor id=', id);
        const res = await getDoctorByIdPublic(id);
        // eslint-disable-next-line no-console
        console.log('DoctorDetail: API response', res && res.data ? res.data : res);
        if (!mounted) return;
        // Handle both { doctor: {...} } and direct {...} responses
        const doctorData = res.data?.doctor || res.data || null;
        // eslint-disable-next-line no-console
        console.log('DoctorDetail: parsed doctorData', doctorData);
        setDoctor(doctorData);
        
        // Load reviews
        try {
          const reviewsRes = await fetch(`${BASE_URL}/reviews/doctor/${id}`);
          const reviewsData = await reviewsRes.json();
          if (!mounted) return;
          if (reviewsData.data) {
            setReviews(reviewsData.data);
          }
        } catch (err) {
          console.warn('Could not load reviews:', err);
        }

        // Load review stats
        try {
          const statsRes = await fetch(`${BASE_URL}/reviews/doctor/${id}/stats`);
          const statsData = await statsRes.json();
          if (!mounted) return;
          if (statsData.data) {
            setReviewStats(statsData.data);
          }
        } catch (err) {
          console.warn('Could not load review stats:', err);
        }

        // Load user's review if authenticated
        if (isAuthenticated) {
          try {
            const userReviewRes = await fetch(`${BASE_URL}/reviews/doctor/${id}/my-review`, {
              headers: {
                Authorization: `Bearer ${localStorage.getItem('token')}`,
              },
            });
            if (!mounted) return;
            const userReviewData = await userReviewRes.json();
            if (userReviewData.data) {
              setUserReview(userReviewData.data);
              setRatingValue(userReviewData.data.rating);
              setReviewComment(userReviewData.data.comment || '');
            }
          } catch (err) {
            console.warn('Could not load user review:', err);
          }
        }

        // try load schedules only when authenticated; avoid triggering global 401 redirect
        if (isAuthenticated) {
          try {
            const sres = await getDoctorSchedules(id);
            if (!mounted) return;
            setSchedules(sres.data?.schedules || sres.data || []);
          } catch (err) {
            // ignore schedules if not available or other error
            // eslint-disable-next-line no-console
            console.warn('Could not load schedules (auth):', err);
          }
        } else {
          // not authenticated: do not call protected schedules endpoint
          // leave schedules empty and show public message
        }
      } catch (err) {
        console.error('Load doctor error', err);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => { mounted = false; };
  }, [id, isAuthenticated]);

  if (loading) return <div className="p-6">Đang tải thông tin bác sĩ...</div>;
  if (!doctor) return <div className="p-6">Không tìm thấy bác sĩ.</div>;

  const handleSubmitReview = async () => {
    if (!isAuthenticated) {
      navigate(`/login?returnUrl=${encodeURIComponent(window.location.pathname)}`);
      return;
    }

    if (ratingValue === 0) {
      alert('Vui lòng chọn số sao');
      return;
    }

    setSubmittingReview(true);
    try {
      const response = await fetch(`${BASE_URL}/reviews`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify({
          doctorId: parseInt(id),
          rating: ratingValue,
          comment: reviewComment,
        }),
      });

      const data = await response.json();
      if (data.success) {
        setUserReview(data.data);
        setShowRatingModal(false);
        // Reload reviews
        const reviewsRes = await fetch(`${BASE_URL}/reviews/doctor/${id}`);
        const reviewsData = await reviewsRes.json();
        if (reviewsData.data) {
          setReviews(reviewsData.data);
        }
        // Reload stats
        const statsRes = await fetch(`${BASE_URL}/reviews/doctor/${id}/stats`);
        const statsData = await statsRes.json();
        if (statsData.data) {
          setReviewStats(statsData.data);
        }
        alert('Đánh giá của bạn đã được gửi thành công!');
      } else {
        alert('Lỗi: ' + data.message);
      }
    } catch (error) {
      console.error('Error submitting review:', error);
      alert('Lỗi khi gửi đánh giá');
    } finally {
      setSubmittingReview(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-start gap-6">
          <div className="w-28 h-28 bg-gray-100 rounded-full flex items-center justify-center overflow-hidden">
            {(() => {
              const avatar = doctor.avatar_url || doctor.user?.avatar_url || doctor.user?.avatar || null;
              return avatar ? (
                <img src={avatar} alt={doctor.user?.full_name || doctor.name} className="w-full h-full object-cover" />
              ) : (
                <div className="text-4xl">👨‍⚕️</div>
              );
            })()}
          </div>
          <div className="flex-1">
            <h1 className="text-2xl font-bold">{doctor.user?.full_name || doctor.name || doctor.fullName || 'Bác sĩ'}</h1>
            
            {/* Specialty & Basic Info */}
            <div className="mt-3 space-y-2">
              {(doctor.specialty || (doctor.specialties && doctor.specialties.length > 0)) && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700">Chuyên khoa:</span>
                  <span className="ml-2 text-gray-600">{doctor.specialty || (doctor.specialties && doctor.specialties.join(', ')) || ''}</span>
                </div>
              )}
              
              {/* Gender */}
              {doctor.gender && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700">Giới tính:</span>
                  <span className="ml-2 text-gray-600">
                    {doctor.gender === 'male' ? 'Nam' : doctor.gender === 'female' ? 'Nữ' : doctor.gender}
                  </span>
                </div>
              )}
              
              {/* Date of Birth (Năm sinh) */}
              {doctor.user?.dob && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700">Năm sinh:</span>
                  <span className="ml-2 text-gray-600">
                    {new Date(doctor.user.dob).getFullYear()}
                  </span>
                </div>
              )}
              
              {/* License Number */}
              {doctor.license_number && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700">Số chứng chỉ:</span>
                  <span className="ml-2 text-gray-600">{doctor.license_number}</span>
                </div>
              )}
              
              {/* Address */}
              {doctor.address && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700">Địa chỉ làm việc:</span>
                  <span className="ml-2 text-gray-600">{doctor.address}</span>
                </div>
              )}
              
              {/* Consultation Fee */}
              {doctor.consultation_fee && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700">Phí khám:</span>
                  <span className="ml-2 text-gray-600">{Number(doctor.consultation_fee).toLocaleString('vi-VN')} VND</span>
                </div>
              )}
              
              {/* Rating */}
              {doctor.rating && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700">Đánh giá:</span>
                  <span className="ml-2 text-yellow-500">★ {Number(doctor.rating).toFixed(1)}/5</span>
                </div>
              )}
            </div>

            {/* Bio/Experience */}
            {doctor.bio && (
              <div className="mt-4 p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-700"><strong>Kinh nghiệm & Mô tả:</strong></p>
                <p className="text-sm text-gray-600 mt-2">{doctor.bio}</p>
              </div>
            )}

            <div className="mt-4 flex gap-3">
              <button
                onClick={() => {
                  if (!isAuthenticated) {
                    // redirect to login with returnUrl pointing to booking form (include doctorId)
                    navigate(`/login?returnUrl=${encodeURIComponent(`/patient/appointments/book?doctorId=${id}`)}`);
                    return;
                  }
                  // redirect to appointment booking page
                  navigate(`/patient/appointments/book?doctorId=${id}`);
                }}
                className="px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700"
              >
                Đặt lịch
              </button>
              <button
                onClick={() => {
                  if (!isAuthenticated) {
                    navigate(`/login?returnUrl=${encodeURIComponent(window.location.pathname)}`);
                    return;
                  }
                  setShowRatingModal(true);
                }}
                className="px-4 py-2 border border-yellow-500 text-yellow-600 rounded hover:bg-yellow-50 flex items-center gap-2"
              >
                <Star size={18} />
                {userReview ? 'Chỉnh sửa đánh giá' : 'Đánh giá bác sĩ'}
              </button>
            </div>
          </div>
        </div>

        <div className="mt-6">
          <h3 className="font-semibold">Thời gian làm việc</h3>
          {schedules.length === 0 ? (
            <div className="text-sm text-gray-600">Thời gian làm việc chưa công khai.</div>
          ) : (
            <ul className="mt-2 space-y-1 text-sm text-gray-700">
              {schedules.map((s, idx) => (
                <li key={idx}>{s.day} — {s.start_time} đến {s.end_time}</li>
              ))}
            </ul>
          )}
        </div>

        {/* Review Stats Section */}
        {reviewStats && (
          <div className="mt-6 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg">
            <h3 className="font-semibold mb-3">Đánh giá từ bệnh nhân ({reviewStats.total_reviews})</h3>
            
            <div className="flex items-center gap-4 mb-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-yellow-600">{reviewStats.average_rating}</div>
                <div className="flex justify-center text-yellow-500">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <Star
                      key={star}
                      size={16}
                      fill={star <= Math.round(reviewStats.average_rating) ? 'currentColor' : 'none'}
                    />
                  ))}
                </div>
                <div className="text-xs text-gray-600 mt-1">({reviewStats.total_reviews} đánh giá)</div>
              </div>

              {/* Rating Distribution */}
              <div className="flex-1">
                {[5, 4, 3, 2, 1].map((rating) => (
                  <div key={rating} className="flex items-center gap-2 text-sm mb-1">
                    <span className="w-6">{rating}★</span>
                    <div className="flex-1 bg-gray-200 rounded h-2">
                      <div
                        className="bg-yellow-500 h-2 rounded"
                        style={{
                          width: `${reviewStats.total_reviews > 0
                            ? (reviewStats.rating_distribution[rating] / reviewStats.total_reviews) * 100
                            : 0
                          }%`,
                        }}
                      />
                    </div>
                    <span className="w-8 text-gray-600">
                      {reviewStats.rating_distribution[rating]}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Reviews List */}
        {reviews.length > 0 && (
          <div className="mt-6">
            <h3 className="font-semibold mb-4">Nhận xét gần đây ({reviews.length})</h3>
            <div className="space-y-4">
              {reviews.slice(0, 5).map((review) => (
                <div key={review.id} className="p-4 border rounded-lg bg-gray-50">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <p className="font-semibold text-sm">
                        {review.patient?.user?.full_name || 'Bệnh nhân'}
                      </p>
                      <p className="text-xs text-gray-500">
                        {new Date(review.created_at).toLocaleDateString('vi-VN')}
                      </p>
                    </div>
                    <div className="flex gap-1">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <Star
                          key={star}
                          size={14}
                          fill={star <= review.rating ? 'currentColor' : 'none'}
                          className={star <= review.rating ? 'text-yellow-500' : 'text-gray-300'}
                        />
                      ))}
                    </div>
                  </div>
                  {review.comment && (
                    <p className="text-sm text-gray-700">{review.comment}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Rating Modal */}
        {showRatingModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
              <h2 className="text-xl font-bold mb-4">Đánh giá bác sĩ {doctor.user?.full_name}</h2>

              {/* Star Rating */}
              <div className="mb-4">
                <p className="text-sm font-semibold mb-3">Xếp hạng:</p>
                <div className="flex justify-center gap-2">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      onClick={() => setRatingValue(star)}
                      onMouseEnter={() => setHoverRating(star)}
                      onMouseLeave={() => setHoverRating(0)}
                      className="transition-transform hover:scale-110"
                    >
                      <Star
                        size={32}
                        fill={star <= (hoverRating || ratingValue) ? 'currentColor' : 'none'}
                        className={
                          star <= (hoverRating || ratingValue)
                            ? 'text-yellow-500'
                            : 'text-gray-300'
                        }
                      />
                    </button>
                  ))}
                </div>
              </div>

              {/* Comment */}
              <div className="mb-4">
                <label className="text-sm font-semibold block mb-2">Nhận xét (tùy chọn):</label>
                <textarea
                  value={reviewComment}
                  onChange={(e) => setReviewComment(e.target.value)}
                  placeholder="Chia sẻ trải nghiệm của bạn với bác sĩ này..."
                  className="w-full p-2 border rounded text-sm"
                  rows={4}
                />
              </div>

              {/* Buttons */}
              <div className="flex gap-2">
                <button
                  onClick={() => setShowRatingModal(false)}
                  className="flex-1 px-4 py-2 border rounded hover:bg-gray-50"
                >
                  Hủy
                </button>
                <button
                  onClick={handleSubmitReview}
                  disabled={submittingReview || ratingValue === 0}
                  className="flex-1 px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:bg-gray-400"
                >
                  {submittingReview ? 'Đang gửi...' : 'Gửi đánh giá'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
