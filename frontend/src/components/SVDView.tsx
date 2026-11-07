import React, { useState, useEffect } from 'react';
import { Info, ArrowRight } from 'lucide-react';
import type { ImageMetadata, SVDAnalysisData } from '../types';
import { getSVDAnalysis } from '../services/api';
import { ScientificPlot } from './ScientificPlot';

interface SVDViewProps {
  metadata: ImageMetadata;
  darkMode: boolean;
  onNavigateToReconstruction: () => void;
}

export const SVDView: React.FC<SVDViewProps> = ({
  metadata,
  darkMode,
  onNavigateToReconstruction,
}) => {
  const [svdData, setSvdData] = useState<SVDAnalysisData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [scaleMode, setScaleMode] = useState<'log' | 'linear'>('log');

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    getSVDAnalysis(metadata.image_id)
      .then((data) => {
        if (isMounted) {
          setSvdData(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error(err);
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [metadata.image_id]);

  if (loading || !svdData) {
    return (
      <div className="flex items-center justify-center p-12 text-slate-400 font-mono text-sm">
        Computing Singular Value Decomposition A = UΣVᵀ via economy LAPACK driver...
      </div>
    );
  }

  const indices = svdData.singular_values.map((_, i) => i + 1);

  const spectrumTraces: any[] = [
    {
      x: indices,
      y: svdData.singular_values,
      type: 'scatter',
      mode: 'lines',
      name: 'Singular Values (σᵢ)',
      line: { color: '#0284c7', width: 2.5 },
    },
  ];

  const energyTraces: any[] = [
    {
      x: indices,
      y: svdData.cumulative_energy_percent,
      type: 'scatter',
      mode: 'lines',
      name: 'Cumulative Energy %',
      line: { color: '#10b981', width: 2.5 },
      fill: 'tozeroy',
      fillcolor: darkMode ? 'rgba(16, 185, 129, 0.08)' : 'rgba(16, 185, 129, 0.1)',
    },
  ];

  const commonLayout = (yTitle: string, isLog: boolean): any => ({
    autosize: true,
    height: 320,
    margin: { l: 55, r: 25, t: 25, b: 45 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: {
      title: { text: 'Singular Component Index i', font: { size: 11, color: darkMode ? '#94a3b8' : '#64748b' } },
      tickfont: { color: darkMode ? '#94a3b8' : '#64748b' },
      gridcolor: darkMode ? '#1e293b' : '#f1f5f9',
    },
    yaxis: {
      title: { text: yTitle, font: { size: 11, color: darkMode ? '#94a3b8' : '#64748b' } },
      type: isLog ? 'log' : 'linear',
      tickfont: { color: darkMode ? '#94a3b8' : '#64748b' },
      gridcolor: darkMode ? '#1e293b' : '#f1f5f9',
    },
  });

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
            Singular Value Decomposition (SVD): A = U Σ Vᵀ
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Fundamental matrix factorization into orthogonal spatial bases U, V and non-negative spectral energies Σ.
          </p>
        </div>

        <button
          onClick={onNavigateToReconstruction}
          className="flex items-center space-x-1.5 text-xs font-semibold px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-700 text-white transition-colors shadow-sm"
        >
          <span>Reconstruct Image with k Components</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-1 shadow-sm">
          <div className="text-xs text-slate-500 dark:text-slate-400">Total Singular Values</div>
          <div className="font-mono text-2xl font-bold text-sky-600 dark:text-sky-400">
            {svdData.total_components}
          </div>
          <div className="text-[10px] text-slate-400 font-mono">
            rank = min({svdData.rows}, {svdData.cols})
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-1 shadow-sm">
          <div className="text-xs text-slate-500 dark:text-slate-400">Largest Singular Value (σ₁)</div>
          <div className="font-mono text-2xl font-bold text-slate-900 dark:text-white">
            {svdData.singular_values[0]?.toFixed(1)}
          </div>
          <div className="text-[10px] text-slate-400 font-mono">
            matrix spectral 2-norm ||A||₂
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-1 shadow-sm">
          <div className="text-xs text-slate-500 dark:text-slate-400">99% Energy Rank</div>
          <div className="font-mono text-2xl font-bold text-emerald-600 dark:text-emerald-400">
            {svdData.effective_rank}
          </div>
          <div className="text-[10px] text-slate-400 font-mono">
            {((svdData.effective_rank / svdData.total_components) * 100).toFixed(1)}% of total components
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-1 shadow-sm">
          <div className="text-xs text-slate-500 dark:text-slate-400">Condition Number κ(A)</div>
          <div className="font-mono text-lg font-bold text-slate-900 dark:text-white mt-1">
            {svdData.condition_number.toExponential(2)}
          </div>
          <div className="text-[10px] text-slate-400 font-mono">
            ratio σ_max / σ_min
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-2 shadow-sm">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white flex items-center space-x-2">
          <Info className="w-4 h-4 text-sky-500" />
          <span>Geometric & Algebraic Interpretation</span>
        </h3>
        <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-sans">
          {svdData.explanation}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-3 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2">
            <span className="text-xs font-semibold text-slate-900 dark:text-white">
              Singular Values Spectrum Decay (Scree Plot)
            </span>
            <div className="flex space-x-1">
              <button
                onClick={() => setScaleMode('log')}
                className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                  scaleMode === 'log'
                    ? 'bg-sky-600 text-white font-medium'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                }`}
              >
                Log
              </button>
              <button
                onClick={() => setScaleMode('linear')}
                className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                  scaleMode === 'linear'
                    ? 'bg-sky-600 text-white font-medium'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                }`}
              >
                Linear
              </button>
            </div>
          </div>
          <ScientificPlot data={spectrumTraces} layout={commonLayout('Singular Value σᵢ', scaleMode === 'log')} />
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-3 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2">
            <span className="text-xs font-semibold text-slate-900 dark:text-white">
              Cumulative Spectral Energy E(k) = (∑ σᵢ² / ∑ σ_all²) %
            </span>
            <span className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400 font-semibold">
              E({svdData.effective_rank}) ≥ 99.0%
            </span>
          </div>
          <ScientificPlot data={energyTraces} layout={commonLayout('Cumulative Energy (%)', false)} />
        </div>
      </div>
    </div>
  );
};
