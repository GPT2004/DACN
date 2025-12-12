import React from 'react';
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { formatCurrency, formatNumber } from '../../utils/helpers';

export default function LineChart({
  data = [],
  lines = [],
  xKey = 'name',
  height = 300,
  showGrid = true,
  showLegend = true,
  showTooltip = true,
  formatYAxis,
  formatTooltip,
  className = '',
}) {
  const defaultFormatYAxis = (value) => {
    if (formatYAxis) return formatYAxis(value);
    return formatNumber(value);
  };

  const defaultFormatTooltip = (value, name) => {
    if (formatTooltip) return formatTooltip(value, name);
    return [formatNumber(value), name];
  };

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <RechartsLineChart data={data}>
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />}
          
          <XAxis
            dataKey={xKey}
            stroke="#666"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#e0e0e0' }}
          />
          
          <YAxis
            stroke="#666"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#e0e0e0' }}
            tickFormatter={defaultFormatYAxis}
          />
          
          {showTooltip && (
            <Tooltip
              formatter={defaultFormatTooltip}
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e0e0e0',
                borderRadius: '8px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              }}
            />
          )}
          
          {showLegend && <Legend />}
          
          {lines.map((line, index) => (
            <Line
              key={line.dataKey || index}
              type="monotone"
              dataKey={line.dataKey}
              name={line.name || line.dataKey}
              stroke={line.color || `hsl(${index * 60}, 70%, 50%)`}
              strokeWidth={line.strokeWidth || 2}
              dot={line.dot !== undefined ? line.dot : { r: 3 }}
              activeDot={{ r: 5 }}
            />
          ))}
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
}