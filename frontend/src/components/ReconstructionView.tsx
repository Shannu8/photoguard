import React, { useState, useEffect } from 'react';
import { Sparkles } from 'lucide-react';
import type { ImageMetadata, ReconstructionData } from '../types';
import { reconstructImage } from '../services/api';

interface ReconstructionViewProps {
  metadata: ImageMetadata;
}

export const ReconstructionView: React.FC<ReconstructionViewProps> = ({ metadata }) => {
  const maxK = Math.min(metadata.width, metadata.height);
  const [k, setK] = useState<number>(Math.min(20, maxK));
  const [mode, setMode] = useState<'rgb' | 'grayscale'>('rgb');
  const [reconstruction, setReconstruction] = useState<ReconstructionData | null>(null);

  const triggerReconstruction = async (targetK: number, targetMode: 'rgb' | 'grayscale') => {
    try {
      const data = await reconstructImage(metadata.image_id, targetK, targetMode);
      setReconstruction(data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    triggerReconstruction(k, mode);
  }, [metadata.image_id, k, mode]);

  const presets = [
    { label: 'k = 1', val: 1 },
    { label: 'k = 5', val: 5 },
    { label: 'k = 10', val: 10 },
    { label: 'k = 25', val: 25 },
    { label: 'k = 50', val: 50 },
    { label: 'k = 100', val: 100 },
    { label: '10% rank', val: Math.max(1, Math.round(maxK * 0.1)) },
    { label: '25% rank', val: Math.max(1, Math.round(maxK * 0.25)) },
    { label: 'Full rank', val: maxK },
  ].filter((p) => p.val <= maxK);

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
          Low-Rank Truncated SVD Image Reconstruction: A_k = U_k Σ_k V_kᵀ
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Eckart-Young-Mirsky optimal low-rank approximation minimizing Frobenius error <span className="font-mono">||A - A_k||_F</span>.
        </p>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
              Reconstruction Manifold:
            </span>
            <div className="inline-flex rounded-lg border border-slate-200 dark:border-slate-800 p-0.5 bg-slate-100 dark:bg-slate-950 text-xs font-medium">
              <button
                onClick={() => setMode('rgb')}
                className={`px-3 py-1 rounded-md transition-colors ${
                  mode === 'rgb'
                    ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400'
                }`}
              >
                Multi-Channel RGB (Channel-Wise SVD)
              </button>
              <button
                onClick={() => setMode('grayscale')}
                className={`px-3 py-1 rounded-md transition-colors ${
                  mode === 'grayscale'
                    ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400'
                }`}
              >
                Single-Channel Grayscale (Luma Matrix)
              </button>
            </div>
          </div>

          <div className="text-xs font-mono text-slate-500 dark:text-slate-400">
            Current k: <span className="font-bold text-sky-600 dark:text-sky-400 text-sm">{k}</span> / {maxK} ({((k / maxK) * 100).toFixed(1)}%)
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center space-x-4">
            <span className="text-xs font-mono text-slate-400">k=1</span>
            <input
              type="range"
              min={1}
              max={maxK}
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              className="flex-1 accent-sky-600 cursor-pointer"
            />
            <span className="text-xs font-mono text-slate-400">k={maxK}</span>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className="text-xs text-slate-400 mr-1">Presets:</span>
            {presets.map((p) => (
              <button
                key={p.label}
                onClick={() => setK(p.val)}
                className={`px-2.5 py-1 text-xs font-mono rounded-md border transition-colors ${
                  k === p.val
                    ? 'bg-sky-50 dark:bg-sky-950/80 border-sky-400 text-sky-700 dark:text-sky-300 font-bold'
                    : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {reconstruction && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
            <div className="text-[11px] text-slate-400 uppercase font-mono">Retained Components</div>
            <div className="font-mono text-lg font-bold text-slate-900 dark:text-white mt-0.5">
              {reconstruction.k} / {reconstruction.total_components}
            </div>
            <div className="text-[10px] text-sky-600 font-mono">
              {reconstruction.retained_percentage}% of rank
            </div>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
            <div className="text-[11px] text-slate-400 uppercase font-mono">Energy Preserved</div>
            <div className="font-mono text-lg font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">
              {reconstruction.energy_preserved_percent}%
            </div>
            <div className="text-[10px] text-slate-400 font-mono">
              ∑_1^k σᵢ² / ∑_1^r σᵢ²
            </div>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
            <div className="text-[11px] text-slate-400 uppercase font-mono">Mean Squared Error (MSE)</div>
            <div className="font-mono text-lg font-bold text-slate-900 dark:text-white mt-0.5">
              {reconstruction.mse.toFixed(2)}
            </div>
            <div className="text-[10px] text-slate-400 font-mono">
              1/N ∑ (I - I_k)²
            </div>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
            <div className="text-[11px] text-slate-400 uppercase font-mono">PSNR (Fidelity)</div>
            <div className="font-mono text-lg font-bold text-sky-600 dark:text-sky-400 mt-0.5">
              {reconstruction.psnr.toFixed(2)} dB
            </div>
            <div className="text-[10px] text-slate-400 font-mono">
              20 log₁₀(255 / √MSE)
            </div>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm col-span-2 sm:col-span-1">
            <div className="text-[11px] text-slate-400 uppercase font-mono">Degrees of Freedom</div>
            <div className="font-mono text-lg font-bold text-slate-900 dark:text-white mt-0.5">
              {reconstruction.compression_ratio}×
            </div>
            <div className="text-[10px] text-slate-400 font-mono">
              (H·W) / k(H+W+1)
            </div>
          </div>
        </div>
      )}

      {reconstruction && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-2 shadow-sm text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-sans">
          <div className="font-semibold text-slate-900 dark:text-white flex items-center space-x-1.5">
            <Sparkles className="w-3.5 h-3.5 text-sky-500" />
            <span>Mathematical Interpretation & Visual Differences at Rank k={k}:</span>
          </div>
          <p>{reconstruction.explanation}</p>
        </div>
      )}

      {reconstruction && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2">
              <span className="text-xs font-semibold text-slate-900 dark:text-white">
                Original Image Matrix (A)
              </span>
              <span className="text-[11px] font-mono text-slate-400">100% Rank</span>
            </div>
            <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex items-center justify-center p-2 min-h-[260px]">
              <img
                src={reconstruction.original_data_url}
                alt="Original"
                className="max-h-72 w-auto object-contain rounded"
              />
            </div>
          </div>

          <div className="rounded-xl border border-sky-300 dark:border-sky-800/80 bg-white dark:bg-slate-900 p-4 space-y-3 shadow-sm ring-1 ring-sky-500/20">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2">
              <span className="text-xs font-semibold text-sky-700 dark:text-sky-300">
                Reconstructed Approximation (A_{k})
              </span>
              <span className="text-[11px] font-mono text-sky-600">k = {reconstruction.k}</span>
            </div>
            <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex items-center justify-center p-2 min-h-[260px]">
              <img
                src={reconstruction.reconstructed_data_url}
                alt={`Reconstructed k=${k}`}
                className="max-h-72 w-auto object-contain rounded"
              />
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2">
              <span className="text-xs font-semibold text-rose-600 dark:text-rose-400">
                Absolute Error Map |A - A_{k}|
              </span>
              <span className="text-[10px] font-mono text-slate-400">Inferno Colormap</span>
            </div>
            <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex items-center justify-center p-2 min-h-[260px]">
              <img
                src={reconstruction.difference_data_url}
                alt="Error Heatmap"
                className="max-h-72 w-auto object-contain rounded"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
