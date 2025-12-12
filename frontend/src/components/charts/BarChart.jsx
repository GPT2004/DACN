import React from 'react';
import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { formatNumber } from '../../utils/helpers';

export default function BarChart({
  data = [],
  bars = [],
  xKey = 'name',
  height = 300,
  showGrid = true,
  showLegend = true,
  showTooltip = true,
  formatYAxis,
  formatTooltip,
  stacked = false,
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
        <RechartsBarChart data={data}>
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
          
          {bars.map((bar, index) => (
            <Bar
              key={bar.dataKey || index}
              dataKey={bar.dataKey}
              name={bar.name || bar.dataKey}
              fill={bar.color || `hsl(${index * 60}, 70%, 50%)`}
              radius={bar.radius || [4, 4, 0, 0]}
              stackId={stacked ? 'stack' : undefined}
            />
          ))}
        </RechartsBarChart>
      </ResponsiveContainer>
    </div>
  );
}