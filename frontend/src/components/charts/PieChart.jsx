import React from 'react';
import {
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { formatNumber } from '../../utils/helpers';

const DEFAULT_COLORS = [
  '#3b82f6', '#22c55e', '#eab308', '#ef4444',
  '#8b5cf6', '#ec4899', '#06b6d4', '#f97316',
];

export default function PieChart({
  data = [],
  dataKey = 'value',
  nameKey = 'name',
  height = 300,
  colors = DEFAULT_COLORS,
  showLegend = true,
  showTooltip = true,
  showLabel = true,
  innerRadius = 0,
  outerRadius = 80,
  formatTooltip,
  formatLabel,
  className = '',
}) {
  const defaultFormatTooltip = (value, name) => {
    if (formatTooltip) return formatTooltip(value, name);
    return [formatNumber(value), name];
  };

  const defaultFormatLabel = (entry) => {
    if (formatLabel) return formatLabel(entry);
    const percent = ((entry.value / data.reduce((sum, item) => sum + item[dataKey], 0)) * 100).toFixed(0);
    return `${entry[nameKey]}: ${percent}%`;
  };

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <RechartsPieChart>
          <Pie
            data={data}
            dataKey={dataKey}
            nameKey={nameKey}
            cx="50%"
            cy="50%"
            innerRadius={innerRadius}
            outerRadius={outerRadius}
            label={showLabel ? defaultFormatLabel : false}
            labelLine={showLabel}
          >
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.color || colors[index % colors.length]}
              />
            ))}
          </Pie>
          
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
        </RechartsPieChart>
      </ResponsiveContainer>
    </div>
  );
}