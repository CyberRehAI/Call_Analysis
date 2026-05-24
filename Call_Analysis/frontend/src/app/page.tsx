'use client';

import { useState } from 'react';
import UploadForm from '@/components/UploadForm';
import AnalysisDashboard from '@/components/AnalysisDashboard';
import { HistorySection } from '@/components/HistorySection';
import { AboutSection } from '@/components/AboutSection';
import { apiService } from '@/lib/api';
import toast from 'react-hot-toast';
import { Mic, Brain, TrendingUp, Shield, Zap } from 'lucide-react';
import type { UploadResponse, StatusResponse, CallResults } from '@/lib/types';

export default function Home() {
  const [uploadedFile, setUploadedFile] = useState<UploadResponse | null>(null);
  const [analysisResults, setAnalysisResults] = useState<CallResults | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState<StatusResponse | null>(null);

  const handleUploadSuccess = (fileData: UploadResponse) => {
    setUploadedFile(fileData);
    setAnalysisResults(null);
    setAnalysisStatus(null);
    toast.success('File uploaded successfully!');
  };

  const handleUploadError = (error: Error) => {
    toast.error(`Upload failed: ${error.message}`);
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) {
      toast.error('Please upload a file first');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisStatus({ status: 'processing', progress: 0, call_id: uploadedFile.call_id });

    try {
      await apiService.analyzeCall(uploadedFile.call_id);
      let pollCount = 0;
      const maxPolls = 300;

      const pollStatus = async () => {
        try {
          pollCount++;
          if (pollCount > maxPolls) {
            setIsAnalyzing(false);
            toast.error('Analysis is taking longer than expected. Please check back later.');
            return;
          }

          const statusResponse = await apiService.getStatus(uploadedFile.call_id);
          setAnalysisStatus(statusResponse);

          if (statusResponse.status === 'processing' || statusResponse.status === 'pending') {
            setTimeout(pollStatus, 2000);
          } else if (statusResponse.status === 'completed') {
            try {
              const results = await apiService.getResults(uploadedFile.call_id);
              setAnalysisResults(results);
              setIsAnalyzing(false);
              toast.success('Analysis completed.');
            } catch (error: any) {
              setIsAnalyzing(false);
              toast.error(error?.message || 'Failed to fetch results.');
            }
          } else if (statusResponse.status === 'failed') {
            setIsAnalyzing(false);
            toast.error('Analysis failed. Please try again.');
          } else {
            setTimeout(pollStatus, 2000);
          }
        } catch (error: any) {
          if (pollCount < 10) setTimeout(pollStatus, 2000);
          else {
            setIsAnalyzing(false);
            toast.error('Error checking status.');
          }
        }
      };

      setTimeout(pollStatus, 1000);
    } catch (error: any) {
      setIsAnalyzing(false);
      setAnalysisStatus(null);
      toast.error(error?.message || 'Failed to start analysis.');
    }
  };

  const handleReset = () => {
    setUploadedFile(null);
    setAnalysisResults(null);
    setIsAnalyzing(false);
    setAnalysisStatus(null);
  };

  return (
    <div>
      <section id="home" className="pt-20 scroll-mt-24">
        <div className="container mx-auto px-6 py-16 md:py-24">
          <div className="max-w-3xl">
            <h1 className="text-4xl md:text-5xl font-bold text-slate-100 mb-4 tracking-tight">
              Call Analysis
            </h1>
            <p className="text-lg text-slate-400 mb-12 max-w-xl">
              Upload a call recording. Get sentiment, emotions, and sales likelihood—without the buzzwords.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="flex items-start gap-3 p-4 rounded-lg bg-slate-800/60 border border-slate-700/50">
                <div className="w-9 h-9 rounded-lg bg-teal-500/20 flex items-center justify-center shrink-0">
                  <Mic className="h-4 w-4 text-teal-400" />
                </div>
                <div>
                  <h3 className="font-medium text-slate-200 text-sm">Transcription</h3>
                  <p className="text-slate-500 text-xs mt-0.5">Whisper-based speech-to-text</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-4 rounded-lg bg-slate-800/60 border border-slate-700/50">
                <div className="w-9 h-9 rounded-lg bg-teal-500/20 flex items-center justify-center shrink-0">
                  <Brain className="h-4 w-4 text-teal-400" />
                </div>
                <div>
                  <h3 className="font-medium text-slate-200 text-sm">Sentiment & emotion</h3>
                  <p className="text-slate-500 text-xs mt-0.5">BERT-style analysis</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-4 rounded-lg bg-slate-800/60 border border-slate-700/50">
                <div className="w-9 h-9 rounded-lg bg-teal-500/20 flex items-center justify-center shrink-0">
                  <TrendingUp className="h-4 w-4 text-teal-400" />
                </div>
                <div>
                  <h3 className="font-medium text-slate-200 text-sm">Sales likelihood</h3>
                  <p className="text-slate-500 text-xs mt-0.5">Conversion probability</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="analyze" className="scroll-mt-24 pb-20">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl">
            <div className="card">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-9 h-9 rounded-lg bg-slate-700 flex items-center justify-center">
                  <Mic className="h-4 w-4 text-slate-400" />
                </div>
                <h2 className="text-xl font-semibold text-slate-100">Upload audio</h2>
              </div>
              <UploadForm
                onUploadSuccess={handleUploadSuccess}
                onUploadError={handleUploadError}
                disabled={isAnalyzing}
              />
              {uploadedFile && (
                <div className="mt-6 p-4 rounded-lg bg-teal-500/10 border border-teal-500/30">
                  <div className="flex items-center gap-2 mb-2">
                    <Shield className="h-4 w-4 text-teal-400" />
                    <span className="font-medium text-teal-200 text-sm">File ready</span>
                  </div>
                  <p className="text-slate-400 text-sm">{uploadedFile.filename}</p>
                  <p className="text-slate-500 text-xs mt-1">ID: {uploadedFile.call_id}</p>
                </div>
              )}
            </div>

            <div className="card">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-9 h-9 rounded-lg bg-slate-700 flex items-center justify-center">
                  <Zap className="h-4 w-4 text-slate-400" />
                </div>
                <h2 className="text-xl font-semibold text-slate-100">Run analysis</h2>
              </div>

              {!uploadedFile ? (
                <div className="py-10 text-center">
                  <Brain className="h-10 w-10 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-500 text-sm">Upload an audio file first.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="w-full btn-primary flex items-center justify-center gap-2 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isAnalyzing ? (
                      <>
                        <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Analyzing…
                      </>
                    ) : (
                      <>
                        <Zap className="h-4 w-4" />
                        Run analysis
                      </>
                    )}
                  </button>

                  {analysisStatus && (
                    <div className="p-4 rounded-lg bg-slate-700/50 border border-slate-600/50">
                      <p className="text-slate-400 text-sm mb-2">
                        Status: <span className="capitalize text-slate-300">{analysisStatus.status}</span>
                      </p>
                      {analysisStatus.progress !== undefined && (
                        <>
                          <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                            <div
                              className="bg-teal-500 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${analysisStatus.progress}%` }}
                            />
                          </div>
                          <p className="text-slate-500 text-xs mt-1">{analysisStatus.progress}%</p>
                        </>
                      )}
                    </div>
                  )}

                  <button onClick={handleReset} className="w-full btn-secondary py-2.5">
                    Reset
                  </button>
                </div>
              )}
            </div>
          </div>

          <section id="results" className="scroll-mt-24 mt-12">
            {analysisResults ? (
              <div className="card max-w-6xl">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-9 h-9 rounded-lg bg-teal-500/20 flex items-center justify-center">
                    <TrendingUp className="h-4 w-4 text-teal-400" />
                  </div>
                  <h2 className="text-xl font-semibold text-slate-100">Results</h2>
                </div>
                <AnalysisDashboard results={analysisResults} />
              </div>
            ) : (
              <div className="max-w-6xl rounded-xl border border-dashed border-slate-600 p-12 text-center">
                <p className="text-slate-500 text-sm">Run an analysis to see results here.</p>
              </div>
            )}
          </section>
        </div>
      </section>

      <section id="history" className="scroll-mt-24 py-16">
        <div className="container mx-auto px-6">
          <HistorySection />
        </div>
      </section>

      <section id="about" className="scroll-mt-24 py-16">
        <div className="container mx-auto px-6">
          <AboutSection />
        </div>
      </section>
    </div>
  );
}
