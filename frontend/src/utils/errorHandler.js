import { message } from 'antd';

/**
 * Handle API error
 */
export const handleApiError = (error) => {
  if (!error.response) {
    message.error('Không thể kết nối đến server');
    return;
  }

  const { status, data } = error.response;

  switch (status) {
    case 400:
      message.error(data.message || 'Dữ liệu không hợp lệ');
      break;
    case 401:
      message.error('Phiên đăng nhập hết hạn. Vui lòng đăng nhập lại');
      // Redirect to login
      window.location.href = '/login';
      break;
    case 403:
      message.error('Bạn không có quyền thực hiện thao tác này');
      break;
    case 404:
      message.error('Không tìm thấy dữ liệu');
      break;
    case 409:
      message.error(data.message || 'Dữ liệu đã tồn tại');
      break;
    case 500:
      message.error('Lỗi server. Vui lòng thử lại sau');
      break;
    default:
      message.error(data.message || 'Có lỗi xảy ra. Vui lòng thử lại');
  }

  return Promise.reject(error);
};

/**
 * Handle form validation errors
 */
export const handleValidationErrors = (errors) => {
  if (!errors || !Array.isArray(errors)) return;

  errors.forEach((error) => {
    message.error(error.message);
  });
};

/**
 * Extract error message
 */
export const getErrorMessage = (error) => {
  if (!error) return 'Có lỗi xảy ra';
  
  if (error.response?.data?.message) {
    return error.response.data.message;
  }
  
  if (error.message) {
    return error.message;
  }
  
  return 'Có lỗi xảy ra';
};