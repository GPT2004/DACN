import dayjs from 'dayjs';
import isSameOrBefore from 'dayjs/plugin/isSameOrBefore';
import isSameOrAfter from 'dayjs/plugin/isSameOrAfter';
import isBetween from 'dayjs/plugin/isBetween';
import customParseFormat from 'dayjs/plugin/customParseFormat';

dayjs.extend(isSameOrBefore);
dayjs.extend(isSameOrAfter);
dayjs.extend(isBetween);
dayjs.extend(customParseFormat);

/**
 * Get start of day
 */
export const getStartOfDay = (date) => {
  return dayjs(date).startOf('day');
};

/**
 * Get end of day
 */
export const getEndOfDay = (date) => {
  return dayjs(date).endOf('day');
};

/**
 * Get start of week
 */
export const getStartOfWeek = (date) => {
  return dayjs(date).startOf('week');
};

/**
 * Get end of week
 */
export const getEndOfWeek = (date) => {
  return dayjs(date).endOf('week');
};

/**
 * Get start of month
 */
export const getStartOfMonth = (date) => {
  return dayjs(date).startOf('month');
};

/**
 * Get end of month
 */
export const getEndOfMonth = (date) => {
  return dayjs(date).endOf('month');
};

/**
 * Check if date is between two dates
 */
export const isDateBetween = (date, startDate, endDate) => {
  return dayjs(date).isBetween(startDate, endDate, null, '[]');
};

/**
 * Check if date is same or before
 */
export const isDateSameOrBefore = (date, compareDate) => {
  return dayjs(date).isSameOrBefore(compareDate);
};

/**
 * Check if date is same or after
 */
export const isDateSameOrAfter = (date, compareDate) => {
  return dayjs(date).isSameOrAfter(compareDate);
};

/**
 * Get days between two dates
 */
export const getDaysBetween = (startDate, endDate) => {
  return dayjs(endDate).diff(dayjs(startDate), 'day');
};

/**
 * Get hours between two dates
 */
export const getHoursBetween = (startDate, endDate) => {
  return dayjs(endDate).diff(dayjs(startDate), 'hour');
};

/**
 * Get minutes between two dates
 */
export const getMinutesBetween = (startDate, endDate) => {
  return dayjs(endDate).diff(dayjs(startDate), 'minute');
};

/**
 * Add days to date
 */
export const addDays = (date, days) => {
  return dayjs(date).add(days, 'day');
};

/**
 * Subtract days from date
 */
export const subtractDays = (date, days) => {
  return dayjs(date).subtract(days, 'day');
};

/**
 * Get week day name
 */
export const getWeekDayName = (date) => {
  const days = ['Chủ nhật', 'Thứ hai', 'Thứ ba', 'Thứ tư', 'Thứ năm', 'Thứ sáu', 'Thứ bảy'];
  return days[dayjs(date).day()];
};

/**
 * Get month name
 */
export const getMonthName = (date) => {
  const months = [
    'Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4',
    'Tháng 5', 'Tháng 6', 'Tháng 7', 'Tháng 8',
    'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12',
  ];
  return months[dayjs(date).month()];
};

/**
 * Get date range for calendar
 */
export const getCalendarDateRange = (date) => {
  const start = dayjs(date).startOf('month').startOf('week');
  const end = dayjs(date).endOf('month').endOf('week');
  return { start, end };
};

/**
 * Get all dates in month
 */
export const getDatesInMonth = (date) => {
  const startOfMonth = dayjs(date).startOf('month');
  const endOfMonth = dayjs(date).endOf('month');
  const dates = [];
  
  let current = startOfMonth;
  while (current.isSameOrBefore(endOfMonth)) {
    dates.push(current);
    current = current.add(1, 'day');
  }
  
  return dates;
};

/**
 * Check if date is weekend
 */
export const isWeekend = (date) => {
  const day = dayjs(date).day();
  return day === 0 || day === 6;
};

/**
 * Get next working day
 */
export const getNextWorkingDay = (date) => {
  let nextDay = dayjs(date).add(1, 'day');
  while (isWeekend(nextDay)) {
    nextDay = nextDay.add(1, 'day');
  }
  return nextDay;
};

/**
 * Parse time string to dayjs
 */
export const parseTime = (timeString) => {
  return dayjs(timeString, 'HH:mm');
};

/**
 * Format time range
 */
export const formatTimeRange = (startTime, endTime) => {
  return `${startTime} - ${endTime}`;
};

/**
 * Check if time is between two times
 */
export const isTimeBetween = (time, startTime, endTime) => {
  const current = parseTime(time);
  const start = parseTime(startTime);
  const end = parseTime(endTime);
  return current.isSameOrAfter(start) && current.isSameOrBefore(end);
};

/**
 * Disable past dates (for DatePicker)
 */
export const disablePastDates = (current) => {
  return current && current < dayjs().startOf('day');
};

/**
 * Disable future dates (for DatePicker)
 */
export const disableFutureDates = (current) => {
  return current && current > dayjs().endOf('day');
};

/**
 * Disable weekends (for DatePicker)
 */
export const disableWeekends = (current) => {
  return current && isWeekend(current);
};