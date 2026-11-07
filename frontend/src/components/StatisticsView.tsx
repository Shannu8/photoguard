import React, { useState, useEffect } from 'react';
import { BarChart2 } from 'lucide-react';
import type { ImageMetadata, ImageStatistics } from '../types';
import { getImageStatistics } from '../services/api';
import { ScientificPlot } from './ScientificPlot';

interface StatisticsViewProps {
  metadata: ImageMetadata;
  darkMode: boolean;
}

export const StatisticsView: React.FC<StatisticsViewProps> = ({ metadata, darkMode }) => {
  const [stats, setStats] = useState<ImageStatistics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedChannel, setSelectedChannel] = useState<'all' | 'gray' | 'red' | 'green' | 'blue'>('all');
  const [logScale, setLogScale] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    getImageStatistics(metadata.image_id)
      .then((data) => {
        if (isMounted) {
          setStats(data);
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

  if (loading || !stats) {
    return (
      <div className="flex items-center justify-center p-12 text-slate-400 font-mono text-sm">
        Computing statistical moments and 256-bin channel histograms...
      </div>
    );
  }

  const binIndices = Array.from({ length: 256 }, (_, i) => i);
  const traces: any[] = [];

  const themeColors = {
    gray: darkMode ? '#cbd5e1' : '#475569',
    red: '#ef4444',
    green: '#10b981',
    blue: '#3b82f6',
  };

  if (selectedChannel === 'all' || selectedChannel === 'red') {
    traces.push({
      x: binIndices,
      y: stats.red.histogram,
      type: 'scatter',
      mode: 'lines',
      name: 'Red Channel',
      line: { color: themeColors.red, width: 2 },
      fill: selectedChannel === 'red' ? 'tozeroy' : 'none',
      fillcolor: 'rgba(239, 68, 68, 0.1)',
    });
  }

  if (selectedChannel === 'all' || selectedChannel === 'green') {
    traces.push({
      x: binIndices,
      y: stats.green.histogram,
      type: 'scatter',
      mode: 'lines',
      name: 'Green Channel',
      line: { color: themeColors.green, width: 2 },
      fill: selectedChannel === 'green' ? 'tozeroy' : 'none',
      fillcolor: 'rgba(16, 185, 129, 0.1)',
    });
  }

  if (selectedChannel === 'all' || selectedChannel === 'blue') {
    traces.push({
      x: binIndices,
      y: stats.blue.histogram,
      type: 'scatter',
      mode: 'lines',
      name: 'Blue Channel',
      line: { color: themeColors.blue, width: 2 },
      fill: selectedChannel === 'blue' ? 'tozeroy' : 'none',
      fillcolor: 'rgba(59, 130, 246, 0.1)',
    });
  }

  if (selectedChannel === 'all' || selectedChannel === 'gray') {
    traces.push({
      x: binIndices,
      y: stats.grayscale.histogram,
      type: 'scatter',
      mode: 'lines',
      name: 'Grayscale (Luma)',
      line: { color: themeColors.gray, width: 2, dash: selectedChannel === 'all' ? 'dash' : 'solid' },
      fill: selectedChannel === 'gray' ? 'tozeroy' : 'none',
      fillcolor: darkMode ? 'rgba(203, 213, 225, 0.1)' : 'rgba(71, 85, 105, 0.1)',
    });
  }

  const plotLayout: any = {
    autosize: true,
    height: 380,
    margin: { l: 50, r: 25, t: 30, b: 45 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: {
      title: { text: 'Pixel Intensity Level [0 .. 255]', font: { size: 12, color: darkMode ? '#94a3b8' : '#64748b' } },
      range: [0, 255],
      tickfont: { color: darkMode ? '#94a3b8' : '#64748b' },
      gridcolor: darkMode ? '#1e293b' : '#f1f5f9',
    },
    yaxis: {
      title: { text: 'Pixel Frequency (Count)', font: { size: 12, color: darkMode ? '#94a3b8' : '#64748b' } },
      type: logScale ? 'log' : 'linear',
      tickfont: { color: darkMode ? '#94a3b8' : '#64748b' },
      gridcolor: darkMode ? '#1e293b' : '#f1f5f9',
    },
    legend: {
      orientation: 'h',
      x: 0.1,
      y: 1.15,
      font: { color: darkMode ? '#e2e8f0' : '#1e293b' },
    },
  };

  const channelCards = [
    { name: 'Grayscale (Luma)', data: stats.grayscale, color: 'border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-200' },
    { name: 'Red Channel (R)', data: stats.red, color: 'border-red-300 dark:border-red-900/60 text-red-600 dark:text-red-400' },
    { name: 'Green Channel (G)', data: stats.green, color: 'border-emerald-300 dark:border-emerald-900/60 text-emerald-600 dark:text-emerald-400' },
    { name: 'Blue Channel (B)', data: stats.blue, color: 'border-blue-300 dark:border-blue-900/60 text-blue-600 dark:text-blue-400' },
  ];

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
          Statistical Image Analysis & Intensity Histograms
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          First-order statistical moments, variance, and discrete probability distributions across individual spectral channels.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {channelCards.map((c) => (
          <div
            key={c.name}
            className={`rounded-xl border bg-white dark:bg-slate-900 p-4 space-y-3 shadow-sm ${c.color}`}
          >
            <div className="text-xs font-semibold uppercase tracking-wider">{c.name}</div>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div>
                <span className="text-slate-400 block text-[10px]">Min / Max:</span>
                <span className="font-semibold text-slate-900 dark:text-white">
                  {c.data.min.toFixed(0)} / {c.data.max.toFixed(0)}
                </span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px]">Mean (μ):</span>
                <span className="font-semibold text-slate-900 dark:text-white">
                  {c.data.mean.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px]">Median:</span>
                <span className="font-semibold text-slate-900 dark:text-white">
                  {c.data.median.toFixed(1)}
                </span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px]">Std Dev (σ):</span>
                <span className="font-semibold text-slate-900 dark:text-white">
                  {c.data.std.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 dark:border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <BarChart2 className="w-4 h-4 text-sky-500" />
            <span className="text-sm font-semibold text-slate-900 dark:text-white">
              Pixel Intensity Frequency Distribution (256 Bins)
            </span>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <div className="flex items-center space-x-1 border border-slate-200 dark:border-slate-800 rounded-lg p-0.5 bg-slate-100 dark:bg-slate-950">
              {(['all', 'gray', 'red', 'green', 'blue'] as const).map((ch) => (
                <button
                  key={ch}
                  onClick={() => setSelectedChannel(ch)}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                    selectedChannel === ch
                      ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                  }`}
                >
                  {ch.toUpperCase()}
                </button>
              ))}
            </div>

            <label className="flex items-center space-x-1.5 cursor-pointer text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={logScale}
                onChange={(e) => setLogScale(e.target.checked)}
                className="rounded accent-sky-600"
              />
              <span className="font-mono text-xs">Log Scale</span>
            </label>
          </div>
        </div>

        <ScientificPlot data={traces} layout={plotLayout} />
      </div>
    </div>
  );
};
