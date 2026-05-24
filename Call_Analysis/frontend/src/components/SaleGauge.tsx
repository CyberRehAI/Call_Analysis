'use client';

import { RadialBarChart, RadialBar, ResponsiveContainer } from 'recharts';

interface SaleGaugeProps {
  value: number; // 0 to 1
}

const SaleGauge = ({ value }: SaleGaugeProps) => {
  const percentage = Math.round(value * 100);
  
  const data = [
    {
      name: 'Sale Probability',
      value: percentage,
      fill: getGaugeColor(percentage),
    },
  ];

  function getGaugeColor(percentage: number) {
    if (percentage >= 70) return '#22c55e'; // Green
    if (percentage >= 40) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
  }

  return (
    <div className="flex flex-col items-center space-y-4">
      <div className="h-64 w-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="60%"
            outerRadius="90%"
            barSize={20}
            data={data}
            startAngle={180}
            endAngle={0}
          >
            <RadialBar
              dataKey="value"
              cornerRadius={10}
              fill={getGaugeColor(percentage)}
            />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="text-center">
        <div className="text-4xl font-bold text-gray-900 mb-2">
          {percentage}%
        </div>
        <div className="text-lg text-gray-600 mb-1">
          Sale Probability
        </div>
        <div className={`text-sm font-medium ${
          percentage >= 70 ? 'text-green-600' : 
          percentage >= 40 ? 'text-yellow-600' : 
          'text-red-600'
        }`}>
          {percentage >= 70 ? 'High' : percentage >= 40 ? 'Medium' : 'Low'}
        </div>
      </div>
    </div>
  );
};

export default SaleGauge;
