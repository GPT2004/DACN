/**
 * Required field validator
 */
export const required = (message = 'Trường này là bắt buộc') => {
  return {
    required: true,
    message,
  };
};

/**
 * Email validator
 */
export const email = (message = 'Email không hợp lệ') => {
  return {
    type: 'email',
    message,
  };
};

/**
 * Min length validator
 */
export const minLength = (min, message) => {
  return {
    min,
    message: message || `Tối thiểu ${min} ký tự`,
  };
};

/**
 * Max length validator
 */
export const maxLength = (max, message) => {
  return {
    max,
    message: message || `Tối đa ${max} ký tự`,
  };
};

/**
 * Phone number validator (Vietnam)
 */
export const phone = (message = 'Số điện thoại không hợp lệ') => {
  return {
    pattern: /^(0|\+84)[0-9]{9,10}$/,
    message,
  };
};

/**
 * Password validator
 */
export const password = (message = 'Mật khẩu tối thiểu 6 ký tự') => {
  return {
    min: 6,
    message,
  };
};

/**
 * Number validator
 */
export const number = (message = 'Vui lòng nhập số') => {
  return {
    type: 'number',
    message,
  };
};

/**
 * Positive number validator
 */
export const positiveNumber = (message = 'Vui lòng nhập số dương') => {
  return {
    type: 'number',
    min: 0,
    message,
  };
};

/**
 * Date validator
 */
export const date = (message = 'Vui lòng chọn ngày') => {
  return {
    type: 'date',
    message,
  };
};

/**
 * Confirm password validator
 */
export const confirmPassword = (getFieldValue) => {
  return {
    validator: (_, value) => {
      if (!value || getFieldValue('password') === value) {
        return Promise.resolve();
      }
      return Promise.reject(new Error('Mật khẩu xác nhận không khớp'));
    },
  };
};

/**
 * Date range validator
 */
export const dateRange = (message = 'Ngày kết thúc phải sau ngày bắt đầu') => {
  return {
    validator: (_, value) => {
      if (!value || !value[0] || !value[1]) {
        return Promise.resolve();
      }
      if (value[0].isBefore(value[1]) || value[0].isSame(value[1])) {
        return Promise.resolve();
      }
      return Promise.reject(new Error(message));
    },
  };
};

/**
 * File size validator
 */
export const fileSize = (maxSize, message) => {
  return {
    validator: (_, file) => {
      if (!file || file.size <= maxSize) {
        return Promise.resolve();
      }
      const maxSizeMB = maxSize / 1024 / 1024;
      return Promise.reject(
        new Error(message || `Kích thước file tối đa ${maxSizeMB}MB`)
      );
    },
  };
};

/**
 * File type validator
 */
export const fileType = (allowedTypes, message) => {
  return {
    validator: (_, file) => {
      if (!file || allowedTypes.includes(file.type)) {
        return Promise.resolve();
      }
      return Promise.reject(
        new Error(message || `Chỉ chấp nhận file: ${allowedTypes.join(', ')}`)
      );
    },
  };
};

/**
 * URL validator
 */
export const url = (message = 'URL không hợp lệ') => {
  return {
    type: 'url',
    message,
  };
};