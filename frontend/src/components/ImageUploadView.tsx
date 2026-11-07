import React, { useState, useRef } from 'react';
import { UploadCloud, AlertCircle, FileCheck, Sparkles, ShieldCheck } from 'lucide-react';
import type { ImageMetadata } from '../types';
import { uploadImage, loadSampleImage } from '../services/api';

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
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setError(null);
    setLoading(true);
    try {
      const meta = await uploadImage(file);
      onImageLoaded(meta);
    } catch (err: any) {
      setError(err.message || 'Failed to upload image');
    } finally {
      setLoading(false);
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

  const handleSample = async () => {
    setError(null);
    setLoading(true);
    try {
      const meta = await loadSampleImage();
      onImageLoaded(meta);
    } catch (err: any) {
      setError(err.message || 'Failed to load sample image');
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

      {error && (
        <div className="flex items-center space-x-2 p-3.5 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-300 text-sm">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-500" />
          <span>{error}</span>
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
            <UploadCloud className="w-8 h-8" />
          </div>
          <div>
            <span className="font-medium text-slate-800 dark:text-slate-200">
              {loading ? 'Processing image tensor...' : 'Click to select or drag and drop image file'}
            </span>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Supported Formats: PNG, JPEG / JPG, WebP, BMP (Max 20MB)
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2 text-xs text-slate-500 dark:text-slate-400">
          <ShieldCheck className="w-4 h-4 text-emerald-500" />
          <span>Integrity Guard: The original raw byte file is preserved immutable and never altered.</span>
        </div>
        <button
          onClick={handleSample}
          disabled={loading}
          className="flex items-center space-x-1.5 text-xs font-medium px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 transition-colors"
        >
          <Sparkles className="w-3.5 h-3.5 text-sky-500" />
          <span>Load Synthetic Benchmark Image</span>
        </button>
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
