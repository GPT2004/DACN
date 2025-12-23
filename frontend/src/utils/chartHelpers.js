/* eslint-disable no-undef */
/**
 * Generate colors for charts
 */
export const generateColors = (count) => {
  const colors = [
    '#1890ff', '#52c41a', '#faad14', '#f5222d',
    '#722ed1', '#13c2c2', '#eb2f96', '#fa8c16',
    '#2f54eb', '#a0d911', '#fa541c', '#9254de',
  ];
  
  const result = [];
  for (let i = 0; i < count; i++) {
    result.push(colors[i % colors.length]);
  }
  return result;
};

/**
 * Format chart data for line chart
 */
export const formatLineChartData = (data, xKey, yKey) => {
  return data.map(item => ({
    name: item[xKey],
    value: item[yKey],
  }));
};

/**
 * Format chart data for pie chart
 */
export const formatPieChartData = (data, nameKey, valueKey) => {
  return data.map(item => ({
    name: item[nameKey],
    value: item[valueKey],
  }));
};

/**
 * Format chart data for bar chart
 */
export const formatBarChartData = (data, categories, series) => {
  return {
    categories: data.map(item => item[categories]),
    series: series.map(s => ({
      name: s.name,
      data: data.map(item => item[s.dataKey]),
    })),
  };
};

/**
 * Calculate chart total
 */
export const calculateChartTotal = (data, valueKey) => {
  return data.reduce((sum, item) => sum + (item[valueKey] || 0), 0);
};

/**
 * Calculate chart percentage
 */
export const calculateChartPercentage = (data, valueKey) => {
  const total = calculateChartTotal(data, valueKey);
  return data.map(item => ({
    ...item,
    percentage: total > 0 ? ((item[valueKey] / total) * 100).toFixed(1) : 0,
  }));
};

/**
 * Get top N items from chart data
 */
export const getTopNItems = (data, valueKey, n = 5) => {
  return [...data]
    .sort((a, b) => b[valueKey] - a[valueKey])
    .slice(0, n);
};

/**
 * Group chart data by time period
 */
export const groupByTimePeriod = (data, dateKey, valueKey, period = 'day') => {
  const grouped = {};
  
  data.forEach(item => {
    const date = dayjs(item[dateKey]);
    let key;
    
    switch (period) {
      case 'day':
        key = date.format('YYYY-MM-DD');
        break;
      case 'week':
        key = `Week ${date.week()}, ${date.year()}`;
        break;
      case 'month':
        key = date.format('YYYY-MM');
        break;
      case 'year':
        key = date.format('YYYY');
        break;
      default:
        key = date.format('YYYY-MM-DD');
    }
    
    if (!grouped[key]) {
      grouped[key] = 0;
    }
    grouped[key] += item[valueKey];
  });
  
  return Object.keys(grouped).map(key => ({
    period: key,
    value: grouped[key],
  }));
};

/**
 * Calculate growth rate
 */
export const calculateGrowthRate = (current, previous) => {
  if (!previous || previous === 0) return 0;
  return ((current - previous) / previous * 100).toFixed(1);
};