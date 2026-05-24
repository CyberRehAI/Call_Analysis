'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Clock, Users, TrendingUp, Eye } from 'lucide-react';
import { apiService } from '@/lib/api';
import toast from 'react-hot-toast';
import type { CallHistoryItem } from '@/lib/types';

export function HistorySection() {
  const [calls, setCalls] = useState<CallHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const data = await apiService.getHistory();
      setCalls(data || []);
    } catch (error: any) {
      console.error('Error fetching history:', error);
      const errorMessage = error?.message || 'Failed to fetch call history';
      toast.error(errorMessage);
      setCalls([]); // Set empty array on error
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const getSentimentColor = (score: number) => {
    if (score >= 0.6) return 'text-green-600 bg-green-100';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getSentimentLabel = (score: number) => {
    if (score >= 0.6) return 'Positive';
    if (score >= 0.4) return 'Neutral';
    return 'Negative';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold text-gray-100">Call History</h2>
        <button onClick={fetchHistory} className="btn-primary">
          Refresh
        </button>
      </div>

      {calls.length === 0 ? (
        <div className="card text-center py-12">
          <Clock className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-100 mb-2">No calls analyzed yet</h3>
          <p className="text-gray-200 mb-4">
            Upload and analyze your first audio file to see it appear here.
          </p>
          <Link href="/#home" className="btn-primary">
            Analyze First Call
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {calls.map((call) => (
            <div key={call.call_id} className="card hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-4 mb-2">
                    <h3 className="text-lg font-semibold text-gray-100">
                      {call.filename}
                    </h3>
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium ${
                        call.status === 'completed'
                          ? 'bg-green-100 text-green-800'
                          : call.status === 'processing'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}
                    >
                      {call.status}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="flex items-center space-x-2">
                      <Clock className="h-4 w-4 text-gray-400" />
                      <span className="text-gray-600">{formatDate(call.timestamp)}</span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Clock className="h-4 w-4 text-gray-400" />
                      <span className="text-gray-600">{formatDuration(call.duration)}</span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Users className="h-4 w-4 text-gray-400" />
                      <span className="text-gray-600">
                        {call.participants} participants
                      </span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <TrendingUp className="h-4 w-4 text-gray-400" />
                      <span className="text-gray-600">
                        {(call.sale_probability * 100).toFixed(1)}% sale probability
                      </span>
                    </div>

                  </div>

                  <div className="mt-3 flex items-center space-x-4">
                    <div
                      className={`px-3 py-1 rounded-full text-sm font-medium ${getSentimentColor(
                        call.avg_sentiment,
                      )}`}
                    >
                      {getSentimentLabel(call.avg_sentiment)} (
                      {call.avg_sentiment.toFixed(2)})
                    </div>
                  </div>
                </div>

                <div className="ml-4">
                  {call.status === 'completed' ? (
                    <Link
                      href={`/results/${call.call_id}`}
                      className="btn-primary flex items-center space-x-2"
                    >
                      <Eye className="h-4 w-4" />
                      <span>View Results</span>
                    </Link>
                  ) : (
                    <button
                      disabled
                      className="btn-secondary flex items-center space-x-2 opacity-50 cursor-not-allowed"
                    >
                      <Eye className="h-4 w-4" />
                      <span>Processing...</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


