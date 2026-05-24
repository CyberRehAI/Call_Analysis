'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import AnalysisDashboard from '@/components/AnalysisDashboard';
import { apiService } from '@/lib/api';
import type { CallResults } from '@/lib/types';
import { ArrowLeft, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface ResultsPageProps {
  params: {
    callId: string;
  };
}

export default function ResultsPage({ params }: ResultsPageProps) {
  const { callId } = params;
  const router = useRouter();
  const [results, setResults] = useState<CallResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const data = await apiService.getResults(callId);
        setResults(data);
      } catch (error: any) {
        toast.error(error?.message ?? 'Failed to load results');
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [callId]);

  return (
    <div className="pt-24 pb-16 container mx-auto px-6">
      <button
        onClick={() => router.back()}
        className="mb-6 inline-flex items-center gap-2 text-slate-400 hover:text-slate-100 text-sm transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </button>

      {loading && (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-10 w-10 animate-spin text-slate-400" />
        </div>
      )}

      {!loading && !results && (
        <div className="card text-center py-12">
          <p className="text-slate-300 mb-2">No results found for this call.</p>
          <p className="text-slate-500 text-sm">The analysis may have failed or the call ID is invalid.</p>
        </div>
      )}

      {results && (
        <div className="card">
          <div className="mb-6">
            <h1 className="text-xl font-semibold text-slate-100 mb-1">Results</h1>
            <p className="text-sm text-slate-500">Call ID: {results.call_id}</p>
          </div>
          <AnalysisDashboard results={results} />
        </div>
      )}
    </div>
  );
}
