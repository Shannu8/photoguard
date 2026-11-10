import React, { useState, useRef } from 'react';
import { UploadCloud, AlertCircle, FileCheck, Sparkles, ShieldCheck, ServerCrash, RefreshCw } from 'lucide-react';
import type { ImageMetadata } from '../types';
import { uploadImage, loadSampleImage } from '../services/api';

const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp', 'bmp'];
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB

interface ImageUploadViewProps {
  metadata: ImageMetadata | null;
  onImageLoaded: (meta: ImageMetadata) => void;
  onNavigateToRGB: () => void;
}

export const ImageUploadView: React.FC<ImageUploadViewProps> = ({
  metadata,
  onImageLoaded,
  onNavigateToRGB,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errorType, setErrorType] = useState<'network' | 'validation' | 'server' | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setError(null);
    setErrorType(null);

    // 1. Client-side empty file validation
    if (!file || file.size === 0) {
      setError('Selected file is empty (0 bytes). Please choose a valid image file.');
      setErrorType('validation');
      return;
    }

    // 2. Client-side file size validation (20MB)
    if (file.size > MAX_FILE_SIZE) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
      setError(`File exceeds size limit: selected file is ${sizeMB} MB (maximum allowed is 20 MB).`);
      setErrorType('validation');
      return;
    }

    // 3. Client-side format validation
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    const isImageMime = file.type.startsWith('image/');
    if (!ALLOWED_EXTENSIONS.includes(ext) && !isImageMime) {
      setError(`Unsupported image format '.${ext || 'unknown'}'. Supported formats: PNG, JPEG/JPG, WebP, BMP.`);
      setErrorType('validation');
      return;
    }

    setLoading(true);
    try {
      const meta = await uploadImage(file);
      onImageLoaded(meta);
    } catch (err: any) {
      const msg = err.message || 'Image upload failed.';
      setError(msg);
      if (msg.includes('Backend unavailable') || msg.includes('Failed to fetch') || msg.includes('port 8000')) {
        setErrorType('network');
      } else if (msg.includes('Unsupported') || msg.includes('Invalid') || msg.includes('empty')) {
        setErrorType('validation');
      } else {
        setErrorType('server');
      }
    } finally {
      setLoading(false);
      // Reset input value so re-uploading the same file works
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleSample = async (preset: 'harmonic' | 'low_rank_blocks' | 'texture_noise' = 'harmonic') => {
    setError(null);
    setErrorType(null);
    setLoading(true);
    try {
      const meta = await loadSampleImage(preset);
      onImageLoaded(meta);
    } catch (err: any) {
      const msg = err.message || 'Failed to load sample image.';
      setError(msg);
      if (msg.includes('Backend unavailable') || msg.includes('Failed to fetch')) {
        setErrorType('network');
      } else {
        setErrorType('server');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
          Image Ingestion & Mathematical Representation
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Upload an image file to convert raw byte encodings into mathematical tensors <span className="font-mono">I ∈ ℝ^(H×W×C)</span>.
        </p>
      </div>

      {/* Structured Error Display */}
      {error && (
        <div className={`p-4 rounded-xl border space-y-2 ${
          errorType === 'network'
            ? 'bg-amber-50 dark:bg-amber-950/40 border-amber-300 dark:border-amber-800 text-amber-900 dark:text-amber-200'
            : 'bg-rose-50 dark:bg-rose-950/40 border-rose-300 dark:border-rose-900 text-rose-900 dark:text-rose-200'
        }`}>
          <div className="flex items-center space-x-2 text-sm font-semibold">
            {errorType === 'network' ? (
              <ServerCrash className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0" />
            ) : (
              <AlertCircle className="w-5 h-5 text-rose-500 shrink-0" />
            )}
            <span>{errorType === 'network' ? 'Backend Service Offline' : 'Upload Error'}</span>
          </div>
          <p className="text-xs leading-relaxed pl-7">{error}</p>
          {errorType === 'network' && (
            <div className="mt-2 pl-7 text-xs bg-amber-100/60 dark:bg-amber-900/30 p-2.5 rounded-lg border border-amber-200 dark:border-amber-800/50 font-mono">
              <span className="font-semibold block mb-1">To start the backend server, run in PowerShell:</span>
              <code className="text-slate-800 dark:text-slate-100 font-bold">$env:PYTHONPATH='backend'; python -m uvicorn app.main:app --port 8000 --reload</code>
            </div>
          )}
        </div>
      )}

      {/* Upload Zone */}
      <div
        onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
        onDragOver={(e) => { e.preventDefault(); }}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
          dragActive
            ? 'border-sky-500 bg-sky-50/50 dark:bg-sky-950/30'
            : 'border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600 bg-white dark:bg-slate-900/60'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.webp,.bmp,image/png,image/jpeg,image/webp,image/bmp"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFile(e.target.files[0]);
            }
          }}
        />
        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="p-3 rounded-full bg-sky-50 dark:bg-sky-950/80 text-sky-600 dark:text-sky-400">
            {loading ? <RefreshCw className="w-8 h-8 animate-spin" /> : <UploadCloud className="w-8 h-8" />}
          </div>
          <div>
            <span className="font-medium text-slate-800 dark:text-slate-200">
              {loading ? 'Processing & validating image tensor...' : 'Click to select or drag and drop image file'}
            </span>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Supported Formats: PNG, JPEG / JPG, WebP, BMP (Max 20MB)
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400">
          <ShieldCheck className="w-4 h-4 text-emerald-500 shrink-0" />
          <span>Integrity Guard: The original raw byte file is preserved immutable and never altered.</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-400">Benchmarks:</span>
          <button
            onClick={() => handleSample('harmonic')}
            disabled={loading}
            className="flex items-center space-x-1 text-xs font-medium px-2.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 transition-colors disabled:opacity-50"
            title="Load continuous harmonic Fourier waves"
          >
            <Sparkles className="w-3.5 h-3.5 text-sky-500" />
            <span>Harmonic Waves</span>
          </button>
          <button
            onClick={() => handleSample('low_rank_blocks')}
            disabled={loading}
            className="flex items-center space-x-1 text-xs font-medium px-2.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 transition-colors disabled:opacity-50"
            title="Load strictly rank-3 geometric block matrix"
          >
            <Sparkles className="w-3.5 h-3.5 text-emerald-500" />
            <span>Rank-3 Blocks</span>
          </button>
        </div>
      </div>

      {metadata && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 space-y-6 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-4">
            <div className="flex items-center space-x-2">
              <FileCheck className="w-5 h-5 text-emerald-500" />
              <h3 className="font-semibold text-slate-900 dark:text-white">
                Image Matrix Specifications
              </h3>
            </div>
            <button
              onClick={onNavigateToRGB}
              className="text-xs font-medium px-3.5 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-700 text-white transition-colors"
            >
              Inspect RGB Matrix →
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Original Image
              </div>
              <div className="relative rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-950 flex items-center justify-center p-2 min-h-[200px]">
                <img
                  src={metadata.rgb_data_url}
                  alt={metadata.filename}
                  className="max-h-56 max-w-full object-contain rounded shadow-sm"
                />
              </div>
            </div>

            <div className="md:col-span-2 grid grid-cols-2 sm:grid-cols-3 gap-4">
              <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-800">
                <div className="text-xs text-slate-500 dark:text-slate-400">Filename</div>
                <div className="font-mono text-sm font-semibold text-slate-900 dark:text-white truncate mt-1">
                  {metadata.filename}
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-800">
                <div className="text-xs text-slate-500 dark:text-slate-400">Dimensions (H × W)</div>
                <div className="font-mono text-sm font-semibold text-slate-900 dark:text-white mt-1">
                  {metadata.height} × {metadata.width}
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-800">
                <div className="text-xs text-slate-500 dark:text-slate-400">Format & Encoding</div>
                <div className="font-mono text-sm font-semibold text-slate-900 dark:text-white mt-1">
                  {metadata.format} ({metadata.color_mode})
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-800">
                <div className="text-xs text-slate-500 dark:text-slate-400">Channels (C)</div>
                <div className="font-mono text-sm font-semibold text-slate-900 dark:text-white mt-1">
                  {metadata.channels} (R, G, B)
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-800">
                <div className="text-xs text-slate-500 dark:text-slate-400">Total Pixels (N)</div>
                <div className="font-mono text-sm font-semibold text-slate-900 dark:text-white mt-1">
                  {metadata.total_pixels.toLocaleString()}
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-800">
                <div className="text-xs text-slate-500 dark:text-slate-400">Raw Storage Size</div>
                <div className="font-mono text-sm font-semibold text-slate-900 dark:text-white mt-1">
                  {(metadata.file_size_bytes / 1024).toFixed(1)} KB
                </div>
              </div>

              <div className="col-span-full p-4 rounded-lg bg-sky-50/50 dark:bg-sky-950/30 border border-sky-100 dark:border-sky-900/40 text-xs text-slate-600 dark:text-slate-300">
                <span className="font-semibold text-sky-800 dark:text-sky-300">Mathematical Tensor Formulation:</span>{' '}
                The image is represented as a 3rd-order discrete tensor{' '}
                <span className="font-mono font-medium">I ∈ ℝ^({metadata.height} × {metadata.width} × 3)</span>, where each entry{' '}
                <span className="font-mono">I[y, x, c] ∈ [0, 255]</span> represents the intensity of color channel{' '}
                <span className="font-mono">c ∈ &#123;0:Red, 1:Green, 2:Blue&#125;</span> at coordinate row <span className="font-mono">y</span> and column <span className="font-mono">x</span>.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
