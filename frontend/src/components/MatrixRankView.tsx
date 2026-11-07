import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, Info, Scale } from 'lucide-react';
import type { ImageMetadata, MatrixRankData } from '../types';
import { getMatrixRank } from '../services/api';
import { ScientificPlot } from './ScientificPlot';

interface MatrixRankViewProps {
  metadata: ImageMetadata;
  darkMode: boolean;
}

export const MatrixRankView: React.FC<MatrixRankViewProps> = ({ metadata, darkMode }) => {
  const [rankData, setRankData] = useState<MatrixRankData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    getMatrixRank(metadata.image_id)
      .then((data) => {
        if (isMounted) {
          setRankData(data);
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

  if (loading || !rankData) {
    return (
      <div className="flex items-center justify-center p-12 text-slate-400 font-mono text-sm">
        Computing SVD singular spectrum and numerical rank...
      </div>
    );
  }

  const topIndices = rankData.singular_values_top.map((_, i) => `σ_${i + 1}`);
  const bottomIndices = rankData.singular_values_bottom.map((_, i) => `σ_${rankData.max_possible_rank - 10 + i + 1}`);

  const traces: any[] = [
    {
      x: topIndices,
      y: rankData.singular_values_top,
      type: 'bar',
      name: 'Top Singular Values',
      marker: { color: '#0284c7' },
    },
    {
      x: bottomIndices,
      y: rankData.singular_values_bottom,
      type: 'bar',
      name: 'Bottom Singular Values',
      marker: { color: '#94a3b8' },
    },
  ];

  const plotLayout: any = {
    autosize: true,
    height: 300,
    margin: { l: 50, r: 25, t: 25, b: 60 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    yaxis: {
      type: 'log',
      title: { text: 'Singular Value σ (Log Scale)', font: { size: 11, color: darkMode ? '#94a3b8' : '#64748b' } },
      tickfont: { color: darkMode ? '#94a3b8' : '#64748b' },
      gridcolor: darkMode ? '#1e293b' : '#f1f5f9',
    },
    xaxis: {
      tickfont: { color: darkMode ? '#94a3b8' : '#64748b', size: 10 },
      gridcolor: darkMode ? '#1e293b' : '#f1f5f9',
    },
    shapes: [
      {
        type: 'line',
        y0: rankData.numerical_tolerance,
        y1: rankData.numerical_tolerance,
        x0: -0.5,
        x1: 19.5,
        line: { color: '#ef4444', width: 2, dash: 'dot' },
      },
    ],
    annotations: [
      {
        x: 10,
        y: Math.log10(Math.max(1e-12, rankData.numerical_tolerance)),
        text: `Tolerance Cutoff: ${rankData.numerical_tolerance.toExponential(2)}`,
        showarrow: false,
        font: { size: 10, color: '#ef4444' },
      },
    ],
    legend: {
      orientation: 'h',
      x: 0.1,
      y: 1.15,
      font: { color: darkMode ? '#e2e8f0' : '#1e293b' },
    },
  };

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
          Grayscale Matrix Rank & Linear Independence
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Dimensionality of the vector space spanned by the row or column vectors of <span className="font-mono">A ∈ ℝ^({rankData.rows}×{rankData.cols})</span>.
        </p>
      </div>

      <div className="rounded-xl border border-amber-300 dark:border-amber-800/80 bg-amber-50/70 dark:bg-amber-950/20 p-5 space-y-2.5">
        <div className="flex items-center space-x-2 text-amber-900 dark:text-amber-300 text-sm font-semibold">
          <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0" />
          <span>Crucial Educational & Mathematical Principle</span>
        </div>
        <p className="text-xs text-amber-900/90 dark:text-amber-200/90 leading-relaxed">
          {rankData.scientific_notes}
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-1 shadow-sm">
          <div className="text-xs text-slate-500 dark:text-slate-400">Calculated Rank rank(A)</div>
          <div className="font-mono text-2xl font-bold text-sky-600 dark:text-sky-400">
            {rankData.rank}
          </div>
          <div className="text-[10px] text-slate-400 font-mono">
            max theoretical: {rankData.max_possible_rank}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-1 shadow-sm">
          <div className="text-xs text-slate-500 dark:text-slate-400">Rank Status</div>
          <div className="flex items-center space-x-2 mt-1">
            {rankData.is_full_rank ? (
              <>
                <CheckCircle className="w-5 h-5 text-emerald-500" />
                <span className="font-semibold text-sm text-emerald-700 dark:text-emerald-400">
                  Full Rank
                </span>
              </>
            ) : (
              <>
                <Info className="w-5 h-5 text-amber-500" />
                <span className="font-semibold text-sm text-amber-700 dark:text-amber-400">
                  Rank Deficient ({rankData.max_possible_rank - rankData.rank} dims)
                </span>
              </>
            )}
          </div>
          <div className="text-[10px] text-slate-400 font-mono">
            min(H, W) = {rankData.max_possible_rank}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-1 shadow-sm">
          <div className="text-xs text-slate-500 dark:text-slate-400">Numerical SVD Tolerance</div>
          <div className="font-mono text-base font-bold text-slate-900 dark:text-white mt-1">
            {rankData.numerical_tolerance.toExponential(4)}
          </div>
          <div className="text-[10px] text-slate-400 font-mono">
            tol = σ₁ · max(H, W) · ε
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-1 shadow-sm">
          <div className="text-xs text-slate-500 dark:text-slate-400">Matrix Dimensions</div>
          <div className="font-mono text-lg font-bold text-slate-900 dark:text-white mt-1">
            {rankData.rows} × {rankData.cols}
          </div>
          <div className="text-[10px] text-slate-400 font-mono">
            {rankData.rows} row scanlines, {rankData.cols} cols
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-3 shadow-sm">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white flex items-center space-x-2">
          <Scale className="w-4 h-4 text-sky-500" />
          <span>Linear Independence & Algebraic Vector Space Analysis</span>
        </h3>
        <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-sans">
          {rankData.explanation}
        </p>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-4 shadow-sm">
        <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2">
          <span className="text-xs font-semibold text-slate-900 dark:text-white">
            Singular Values Spectrum & Numerical Cutoff (σ &gt; tol)
          </span>
          <span className="text-[11px] font-mono text-slate-400">
            Top 10 vs Bottom 10 Singular Values
          </span>
        </div>
        <ScientificPlot data={traces} layout={plotLayout} />
      </div>
    </div>
  );
};
