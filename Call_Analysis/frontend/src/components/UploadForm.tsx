'use client';

import { useState, useRef } from 'react';
import { Upload, FileAudio, X } from 'lucide-react';
import { apiService } from '@/lib/api';
import toast from 'react-hot-toast';
import type { UploadResponse } from '@/lib/types';

interface UploadFormProps {
  onUploadSuccess: (fileData: UploadResponse) => void;
  onUploadError: (error: Error) => void;
  disabled?: boolean;
}

const UploadForm = ({ onUploadSuccess, onUploadError, disabled }: UploadFormProps) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/mp4'];
  const maxSize = 100 * 1024 * 1024;

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!allowedTypes.includes(file.type)) {
      toast.error('Use .wav, .mp3, or .m4a (max 100MB)');
      return;
    }
    if (file.size > maxSize) {
      toast.error('File must be under 100MB');
      return;
    }
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setUploadProgress(0);
    try {
      const response = await apiService.uploadAudio(selectedFile);
      onUploadSuccess({ ...response, filename: selectedFile.name, size: selectedFile.size });
    } catch {
      onUploadError(new Error('Upload failed'));
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-4">
      <div
        className="border-2 border-dashed border-slate-600 rounded-xl p-6 text-center hover:border-slate-500 transition-colors bg-slate-800/40"
        onClick={() => !selectedFile && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp3,.m4a"
          onChange={handleFileSelect}
          disabled={disabled || isUploading}
          className="hidden"
        />
        {!selectedFile ? (
          <div className="cursor-pointer">
            <div className="w-14 h-14 mx-auto mb-4 rounded-xl bg-slate-700 flex items-center justify-center">
              <Upload className="h-6 w-6 text-slate-400" />
            </div>
            <p className="text-slate-300 font-medium mb-1">Drop audio here or click to browse</p>
            <p className="text-slate-500 text-sm">.wav, .mp3, .m4a — max 100MB</p>
          </div>
        ) : (
          <div className="flex items-center justify-between p-4 rounded-lg bg-slate-700/50 border border-slate-600/50 text-left">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-teal-500/20 flex items-center justify-center shrink-0">
                <FileAudio className="h-5 w-5 text-teal-400" />
              </div>
              <div>
                <p className="font-medium text-slate-200 text-sm">{selectedFile.name}</p>
                <p className="text-slate-500 text-xs">{formatFileSize(selectedFile.size)}</p>
              </div>
            </div>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); handleRemoveFile(); }}
              disabled={disabled || isUploading}
              className="w-8 h-8 rounded-lg bg-slate-600 hover:bg-slate-500 flex items-center justify-center text-slate-400 hover:text-slate-200 disabled:opacity-50 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>

      {selectedFile && (
        <button
          onClick={handleUpload}
          disabled={disabled || isUploading}
          className="w-full btn-primary flex items-center justify-center gap-2 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isUploading ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Uploading…
            </>
          ) : (
            <>
              <Upload className="h-4 w-4" />
              Upload file
            </>
          )}
        </button>
      )}

      {isUploading && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-slate-500">
            <span>Uploading…</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-teal-500 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadForm;
