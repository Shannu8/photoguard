import React from 'react';
import { Sun, Moon, Binary, Sparkles } from 'lucide-react';
import type { ImageMetadata } from '../types';

interface HeaderProps {
  darkMode: boolean;
  setDarkMode: (val: boolean) => void;
  metadata: ImageMetadata | null;
  onLoadSample: () => void;
  loadingSample: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  darkMode,
  setDarkMode,
  metadata,
  onLoadSample,
  loadingSample,
}) => {
  return (
    <header className="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur sticky top-0 z-30 px-6 py-3.5 transition-colors">
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-sky-500/10 dark:bg-sky-500/20 text-sky-600 dark:text-sky-400">
            <Binary className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-semibold text-lg tracking-tight text-slate-900 dark:text-white">
                MathOptic <span className="text-sky-600 dark:text-sky-400 font-normal">| Image Analysis</span>
              </h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-sky-100 dark:bg-sky-950/60 text-sky-800 dark:text-sky-300 border border-sky-300 dark:border-sky-800">
                Phase 1 & 2
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Matrix Decomposition, SVD Spectrum & Linear Algebra Image Laboratory
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {metadata && (
            <div className="hidden md:flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>{metadata.filename}</span>
              <span className="text-slate-400">({metadata.width}×{metadata.height}, {metadata.format})</span>
            </div>
          )}

          <button
            onClick={onLoadSample}
            disabled={loadingSample}
            className="flex items-center space-x-1.5 text-xs font-medium px-3 py-1.5 rounded-md bg-sky-50 hover:bg-sky-100 dark:bg-sky-900/30 dark:hover:bg-sky-900/50 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800 transition-colors disabled:opacity-50"
            title="Load mathematical benchmark harmonic wave image"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>{loadingSample ? 'Loading...' : 'Load Sample Matrix'}</span>
          </button>

          <button
            onClick={() => setDarkMode(!darkMode)}
            className="p-2 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 transition-colors"
            title="Toggle theme"
          >
            {darkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-600" />}
          </button>
        </div>
      </div>
    </header>
  );
};
