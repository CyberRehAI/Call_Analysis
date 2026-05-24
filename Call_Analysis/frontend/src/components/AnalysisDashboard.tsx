'use client';

import { useState } from 'react';
import SentimentChart from './SentimentChart';
import EmotionChart from './EmotionChart';
import SaleGauge from './SaleGauge';
import KeyPhrases from './KeyPhrases';
import { TrendingUp, TrendingDown, Users, Clock, Download, FileText, FileSpreadsheet } from 'lucide-react';
import toast from 'react-hot-toast';
import { apiService } from '@/lib/api';
import type { CallResults } from '@/lib/types';

interface AnalysisDashboardProps {
  results: CallResults;
}

const AnalysisDashboard = ({ results }: AnalysisDashboardProps) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isExporting, setIsExporting] = useState(false);

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  const handleExportPDF = async () => {
    setIsExporting(true);
    try {
      const blob = await apiService.exportPDF(results.call_id);
      downloadBlob(blob, `${results.call_id}_report.pdf`);
      toast.success('PDF report downloaded successfully!');
    } catch (error: any) {
      toast.error(error?.message || 'Failed to export PDF report');
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportJSON = async () => {
    setIsExporting(true);
    try {
      const blob = await apiService.exportJSON(results.call_id);
      downloadBlob(blob, `${results.call_id}_analysis.json`);
      toast.success('JSON data downloaded successfully!');
    } catch (error: any) {
      toast.error(error?.message || 'Failed to export JSON data');
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportCSV = async () => {
    setIsExporting(true);
    try {
      const blob = await apiService.exportCSV(results.call_id);
      downloadBlob(blob, `${results.call_id}_analysis.csv`);
      toast.success('CSV data downloaded successfully!');
    } catch (error: any) {
      toast.error(error?.message || 'Failed to export CSV data');
    } finally {
      setIsExporting(false);
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'sentiment', label: 'Sentiment Analysis' },
    { id: 'emotions', label: 'Emotion Detection' },
    { id: 'phrases', label: 'Key Phrases' },
  ];

  const getSentimentLabel = (score: number) => {
    if (score >= 0.6) return 'Positive';
    if (score >= 0.4) return 'Neutral';
    return 'Negative';
  };

  const getSentimentColor = (score: number) => {
    if (score >= 0.6) return 'text-green-400';
    if (score >= 0.4) return 'text-amber-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Average Sentiment</p>
              <p className={`text-2xl font-bold ${getSentimentColor(results.summary.avg_sentiment)}`}>
                {getSentimentLabel(results.summary.avg_sentiment)}
              </p>
              <p className="text-sm text-slate-500">Score: {results.summary.avg_sentiment.toFixed(2)}</p>
            </div>
            {results.summary.avg_sentiment >= 0.5 ? (
              <TrendingUp className="h-8 w-8 text-green-400" />
            ) : (
              <TrendingDown className="h-8 w-8 text-red-400" />
            )}
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Sale Probability</p>
              <p className="text-2xl font-bold text-teal-400">{(results.sale_probability * 100).toFixed(1)}%</p>
            </div>
            <TrendingUp className="h-8 w-8 text-teal-400" />
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Duration</p>
              <p className="text-2xl font-bold text-slate-100">
                {Math.floor(results.summary.total_duration / 60)}m {results.summary.total_duration % 60}s
              </p>
            </div>
            <Clock className="h-8 w-8 text-slate-400" />
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-400">Participants</p>
              <p className="text-2xl font-bold text-slate-100">{results.summary.participants}</p>
            </div>
            <Users className="h-8 w-8 text-slate-400" />
          </div>
        </div>
      </div>

      <div className="flex flex-wrap justify-between items-center gap-4">
        <nav className="flex gap-1 border-b border-slate-600">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-2 px-3 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-teal-500 text-teal-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <div className="flex gap-2">
          <button onClick={handleExportPDF} disabled={isExporting} className="btn-secondary flex items-center gap-2 text-sm py-2 px-3 disabled:opacity-50">
            <FileText className="h-4 w-4" /> PDF
          </button>
          <button onClick={handleExportCSV} disabled={isExporting} className="btn-secondary flex items-center gap-2 text-sm py-2 px-3 disabled:opacity-50">
            <FileSpreadsheet className="h-4 w-4" /> CSV
          </button>
          <button onClick={handleExportJSON} disabled={isExporting} className="btn-secondary flex items-center gap-2 text-sm py-2 px-3 disabled:opacity-50">
            <Download className="h-4 w-4" /> JSON
          </button>
        </div>
      </div>

      <div className="space-y-6">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-slate-100 mb-4">Sentiment over time</h3>
              <SentimentChart data={results.sentiment_scores} />
            </div>
            <div className="card">
              <h3 className="text-lg font-semibold text-slate-100 mb-4">Emotion distribution</h3>
              <EmotionChart data={results.emotions} />
            </div>
            <div className="card">
              <h3 className="text-lg font-semibold text-slate-100 mb-4">Sale probability</h3>
              <SaleGauge value={results.sale_probability} />
            </div>
          </div>
        )}
        {activeTab === 'sentiment' && (
          <div className="card">
            <h3 className="text-lg font-semibold text-slate-100 mb-4">Sentiment analysis</h3>
            <SentimentChart data={results.sentiment_scores} />
          </div>
        )}
        {activeTab === 'emotions' && (
          <div className="card">
            <h3 className="text-lg font-semibold text-slate-100 mb-4">Emotion detection</h3>
            <EmotionChart data={results.emotions} />
          </div>
        )}
        {activeTab === 'phrases' && (
          <div className="card">
            <h3 className="text-lg font-semibold text-slate-100 mb-4">Key phrases</h3>
            <KeyPhrases
              phrases={{
                positive: results.key_phrases.positive.map((p) => p.phrase),
                negative: results.key_phrases.negative.map((p) => p.phrase),
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisDashboard;
