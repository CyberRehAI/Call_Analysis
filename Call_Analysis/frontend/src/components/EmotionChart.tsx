'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

interface EmotionChartProps {
  data: Record<string, number>;
}

const EmotionChart = ({ data }: EmotionChartProps) => {
  // Transform data for the chart
  const chartData = Object.entries(data).map(([emotion, value]) => ({
    name: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    value: value,
    percentage: ((value / Object.values(data).reduce((a, b) => a + b, 0)) * 100).toFixed(1),
  }));

  // Color palette for emotions (keys must match backend: neutral, happiness, anger, sadness, frustration)
  const COLORS: Record<string, string> = {
    neutral: '#FFFFFF', // gray-500
    happiness: '#22c55e', // green-500
    anger: '#ef4444', // red-500
    sadness: '#3b82f6', // blue-500
    frustration: '#f59e0b', // amber-500
  };

  const getColor = (emotion: string) => {
    return COLORS[emotion.toLowerCase() as keyof typeof COLORS] || '#6b7280';
  };

  const formatTooltip = (value: number, name: string, props: any) => {
    const percentage = props.payload.percentage;
    return [`${value.toFixed(2)} (${percentage}%)`, name];
  };

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percentage }) => `${name} (${percentage}%)`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(entry.name)} />
            ))}
          </Pie>
          <Tooltip formatter={formatTooltip} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default EmotionChart;
