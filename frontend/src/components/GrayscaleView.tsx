import React, { useState } from 'react';
import { Sparkles } from 'lucide-react';
import type { ImageMetadata } from '../types';

interface GrayscaleViewProps {
  metadata: ImageMetadata;
}

export const GrayscaleView: React.FC<GrayscaleViewProps> = ({ metadata }) => {
  const [activeDisplay, setActiveDisplay] = useState<'side_by_side' | 'rgb_only' | 'grayscale_only'>('side_by_side');
  const [testR, setTestR] = useState<number>(200);
  const [testG, setTestG] = useState<number>(100);
  const [testB, setTestB] = useState<number>(50);

  const calcTestY = 0.299 * testR + 0.587 * testG + 0.114 * testB;

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
          RGB-to-Grayscale Transformation & Photopic Luma
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Mathematically transparent linear transformation from 3-channel color manifold to single-channel scalar luminance field.
        </p>
      </div>

      <div className="rounded-xl border border-sky-200 dark:border-sky-900/60 bg-sky-50/50 dark:bg-sky-950/20 p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-sky-100 dark:bg-sky-900 text-sky-800 dark:text-sky-300 font-mono">
              ITU-R BT.601 Standard
            </span>
            <span className="text-xs font-mono text-slate-600 dark:text-slate-300">
              Y = 0.299 · R + 0.587 · G + 0.114 · B
            </span>
          </div>
          <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
            Σ weights = 0.299 + 0.587 + 0.114 = 1.000
          </span>
        </div>

        <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
          The scalar luminance <span className="font-mono font-medium">Y</span> is a weighted inner product of color coordinates with the photopic spectral sensitivity vector of the human visual system. Humans perceive green photons with the highest luminous efficiency (~58.7%), followed by red (~29.9%) and blue (~11.4%).
        </p>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">Display Layout:</span>
          <div className="inline-flex rounded-lg border border-slate-200 dark:border-slate-800 p-0.5 bg-slate-100 dark:bg-slate-900">
            <button
              onClick={() => setActiveDisplay('side_by_side')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${activeDisplay === 'side_by_side'
                  ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setActiveDisplay('rgb_only')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${activeDisplay === 'rgb_only'
                  ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
            >
              Original RGB Only
            </button>
            <button
              onClick={() => setActiveDisplay('grayscale_only')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${activeDisplay === 'grayscale_only'
                  ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                }`}
            >
              Grayscale (Y) Only
            </button>
          </div>
        </div>

        <span className="text-xs font-mono text-slate-400">
          Matrix Size: {metadata.height} × {metadata.width}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {(activeDisplay === 'side_by_side' || activeDisplay === 'rgb_only') && (
          <div className={`rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3 shadow-sm ${activeDisplay === 'rgb_only' ? 'md:col-span-2' : ''
            }`}>
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2">
              <span className="text-xs font-semibold text-slate-900 dark:text-white">
                Original Color Tensor: I ∈ ℝ^{'{'}{metadata.height}×{metadata.width}×3{'}'}
              </span>
              <span className="text-[11px] font-mono text-slate-400">3 Channels (RGB)</span>
            </div>
            <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex items-center justify-center p-2 min-h-[300px]">
              <img
                src={metadata.rgb_data_url}
                alt="Original RGB"
                className="max-h-80 w-auto object-contain rounded"
              />
            </div>
          </div>
        )}

        {(activeDisplay === 'side_by_side' || activeDisplay === 'grayscale_only') && (
          <div className={`rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3 shadow-sm ${activeDisplay === 'grayscale_only' ? 'md:col-span-2' : ''
            }`}>
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2">
              <span className="text-xs font-semibold text-slate-900 dark:text-white">
                Grayscale Matrix: A ∈ ℝ^{'{'}{metadata.height}×{metadata.width}{'}'}
              </span>
              <span className="text-[11px] font-mono text-slate-400">1 Channel (Luminance)</span>
            </div>
            <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-950 flex items-center justify-center p-2 min-h-[300px]">
              <img
                src={metadata.grayscale_data_url}
                alt="Grayscale Matrix"
                className="max-h-80 w-auto object-contain rounded"
              />
            </div>
          </div>
        )}
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-4 shadow-sm">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-sky-500" />
          <h3 className="font-semibold text-sm text-slate-900 dark:text-white">
            Interactive Pixel Linear Projection Calculator
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
          <div>
            <label className="text-xs font-mono text-red-600 dark:text-red-400 flex justify-between">
              <span>Red (R): {testR}</span>
              <span>{(testR / 255).toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={255}
              value={testR}
              onChange={(e) => setTestR(parseInt(e.target.value))}
              className="w-full mt-1 accent-red-500"
            />
          </div>

          <div>
            <label className="text-xs font-mono text-emerald-600 dark:text-emerald-400 flex justify-between">
              <span>Green (G): {testG}</span>
              <span>{(testG / 255).toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={255}
              value={testG}
              onChange={(e) => setTestG(parseInt(e.target.value))}
              className="w-full mt-1 accent-emerald-500"
            />
          </div>

          <div>
            <label className="text-xs font-mono text-blue-600 dark:text-blue-400 flex justify-between">
              <span>Blue (B): {testB}</span>
              <span>{(testB / 255).toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={255}
              value={testB}
              onChange={(e) => setTestB(parseInt(e.target.value))}
              className="w-full mt-1 accent-blue-500"
            />
          </div>

          <div className="p-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/80 flex items-center space-x-3">
            <div
              className="w-10 h-10 rounded-md border border-slate-300 dark:border-slate-600 shrink-0"
              style={{ backgroundColor: `rgb(${testR}, ${testG}, ${testB})` }}
            />
            <div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-mono">Output Luminance (Y)</div>
              <div className="font-mono text-base font-bold text-slate-900 dark:text-white">
                {calcTestY.toFixed(3)}
              </div>
            </div>
          </div>
        </div>

        <pre className="text-xs font-mono p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-300 overflow-x-auto">
          Y = 0.299 · ({testR}) + 0.587 · ({testG}) + 0.114 · ({testB})
          = {(0.299 * testR).toFixed(3)} + {(0.587 * testG).toFixed(3)} + {(0.114 * testB).toFixed(3)}
          = {calcTestY.toFixed(3)} (rounds to integer {Math.round(calcTestY)} in 8-bit [0..255])
        </pre>
      </div>
    </div>
  );
};
