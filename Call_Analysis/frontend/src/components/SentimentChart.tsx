'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface SentimentData {
  timestamp: number;
  score: number;
}

interface SentimentChartProps {
  data: SentimentData[];
}

const SentimentChart = ({ data }: SentimentChartProps) => {
  // Transform data for the chart
  const chartData = data.map((item, index) => ({
    time: `T${index + 1}`,
    sentiment: item.score,
    timestamp: item.timestamp,
  }));

  const formatTooltip = (value: number) => {
    const sentiment = value >= 0.6 ? 'Positive' : value >= 0.4 ? 'Neutral' : 'Negative';
    return [`${sentiment} (${value.toFixed(2)})`, 'Sentiment'];
  };

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="time" 
            stroke="#6b7280"
            fontSize={12}
          />
          <YAxis 
            domain={[0, 1]}
            stroke="#6b7280"
            fontSize={12}
            tickFormatter={(value) => value.toFixed(1)}
          />
          <Tooltip 
            formatter={formatTooltip}
            labelStyle={{ color: '#374151' }}
            contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            }}
          />
          <Line 
            type="monotone" 
            dataKey="sentiment" 
            stroke="#3b82f6" 
            strokeWidth={2}
            dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6, stroke: '#3b82f6', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SentimentChart;
