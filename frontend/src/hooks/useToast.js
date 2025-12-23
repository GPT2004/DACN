/* eslint-disable no-console */
import { useCallback } from 'react';

export const useToast = () => {
  const success = useCallback((message) => {
    // Integration with your toast library
    console.log('Success:', message);
    // toast.success(message);
  }, []);

  const error = useCallback((message) => {
    console.log('Error:', message);
    // toast.error(message);
  }, []);

  const warning = useCallback((message) => {
    console.log('Warning:', message);
    // toast.warning(message);
  }, []);

  const info = useCallback((message) => {
    console.log('Info:', message);
    // toast.info(message);
  }, []);

  return { success, error, warning, info };
};