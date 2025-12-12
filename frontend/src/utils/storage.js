/* eslint-disable no-console */
const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  USER_DATA: 'user_data',
  THEME: 'theme',
  LANGUAGE: 'language',
  SIDEBAR_COLLAPSED: 'sidebar_collapsed',
};

/**
 * Get item from localStorage
 */
export const getStorageItem = (key) => {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : null;
  } catch (error) {
    console.error('Error getting storage item:', error);
    return null;
  }
};

/**
 * Set item to localStorage
 */
export const setStorageItem = (key, value) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error('Error setting storage item:', error);
    return false;
  }
};

/**
 * Remove item from localStorage
 */
export const removeStorageItem = (key) => {
  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error('Error removing storage item:', error);
    return false;
  }
};

/**
 * Clear all localStorage
 */
export const clearStorage = () => {
  try {
    localStorage.clear();
    return true;
  } catch (error) {
    console.error('Error clearing storage:', error);
    return false;
  }
};

// Auth storage helpers
export const getAuthToken = () => getStorageItem(STORAGE_KEYS.AUTH_TOKEN);
export const setAuthToken = (token) => setStorageItem(STORAGE_KEYS.AUTH_TOKEN, token);
export const removeAuthToken = () => removeStorageItem(STORAGE_KEYS.AUTH_TOKEN);

export const getRefreshToken = () => getStorageItem(STORAGE_KEYS.REFRESH_TOKEN);
export const setRefreshToken = (token) => setStorageItem(STORAGE_KEYS.REFRESH_TOKEN, token);
export const removeRefreshToken = () => removeStorageItem(STORAGE_KEYS.REFRESH_TOKEN);

export const getUserData = () => getStorageItem(STORAGE_KEYS.USER_DATA);
export const setUserData = (data) => setStorageItem(STORAGE_KEYS.USER_DATA, data);
export const removeUserData = () => removeStorageItem(STORAGE_KEYS.USER_DATA);

export const clearAuthData = () => {
  removeAuthToken();
  removeRefreshToken();
  removeUserData();
};

export { STORAGE_KEYS };