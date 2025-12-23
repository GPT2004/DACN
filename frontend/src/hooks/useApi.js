/* eslint-disable react-hooks/exhaustive-deps */
// frontend/src/hooks/useApi.js
import { useState, useCallback } from 'react';

export const useApi = (apiFunc) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const execute = useCallback(
    async (...args) => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await apiFunc(...args);
        setData(response.data);
        
        return { success: true, data: response.data };
      } catch (err) {
        const errorMsg = err.response?.data?.message || 'Có lỗi xảy ra';
        setError(errorMsg);
        return { success: false, error: errorMsg };
      } finally {
        setLoading(false);
      }
    },
    [apiFunc]
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
};

// Hook for paginated data
export const usePaginatedApi = (apiFunc) => {
  const [data, setData] = useState([]);
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 10,
    total: 0,
    totalPages: 0,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetch = useCallback(
    async (params = {}) => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await apiFunc({
          page: pagination.page,
          limit: pagination.limit,
          ...params,
        });

        setData(response.data[Object.keys(response.data)[0]] || []);
        setPagination(response.data.pagination || pagination);
        
        return { success: true, data: response.data };
      } catch (err) {
        const errorMsg = err.response?.data?.message || 'Có lỗi xảy ra';
        setError(errorMsg);
        return { success: false, error: errorMsg };
      } finally {
        setLoading(false);
      }
    },
    [apiFunc, pagination.page, pagination.limit]
  );

  const changePage = useCallback((page) => {
    setPagination(prev => ({ ...prev, page }));
  }, []);

  const changePageSize = useCallback((limit) => {
    setPagination(prev => ({ ...prev, limit, page: 1 }));
  }, []);

  return {
    data,
    pagination,
    loading,
    error,
    fetch,
    changePage,
    changePageSize,
  };
};