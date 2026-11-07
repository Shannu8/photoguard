import React from 'react';
import {
  UploadCloud,
  Grid3X3,
  Sliders,
  BarChart2,
  Cpu,
  Layers,
  Sparkles,
  ArrowRightLeft,
} from 'lucide-react';

export type ActiveTab =
  | 'upload'
  | 'rgb'
  | 'grayscale'
  | 'statistics'
  | 'rank'
  | 'eigen'
  | 'svd'
  | 'reconstruction';

interface SidebarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  hasImage: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  hasImage,
}) => {
  const navItems = [
    {
      id: 'upload' as ActiveTab,
      label: 'Image Upload',
      sublabel: 'PNG, JPEG, WebP, BMP',
      icon: UploadCloud,
      phase: 'Phase 1',
    },
    {
      id: 'rgb' as ActiveTab,
      label: 'RGB Matrix & Pixels',
      sublabel: 'I ∈ ℝ^{H×W×3}',
      icon: Grid3X3,
      phase: 'Phase 1',
      requiresImage: true,
    },
    {
      id: 'grayscale' as ActiveTab,
      label: 'Grayscale Conversion',
      sublabel: 'Y = 0.299R + 0.587G + 0.114B',
      icon: ArrowRightLeft,
      phase: 'Phase 1',
      requiresImage: true,
    },
    {
      id: 'statistics' as ActiveTab,
      label: 'Image Statistics',
      sublabel: 'Min, Max, Histograms',
      icon: BarChart2,
      phase: 'Phase 1',
      requiresImage: true,
    },
    {
      id: 'rank' as ActiveTab,
      label: 'Matrix Rank',
      sublabel: 'rank(A), Full-Rank & Tol',
      icon: Layers,
      phase: 'Phase 2',
      requiresImage: true,
    },
    {
      id: 'eigen' as ActiveTab,
      label: 'Eigenvalue Analysis',
      sublabel: 'Av = λv & Gram Spectra',
      icon: Cpu,
      phase: 'Phase 2',
      requiresImage: true,
    },
    {
      id: 'svd' as ActiveTab,
      label: 'SVD Decomposition',
      sublabel: 'A = U Σ Vᵀ & Energy',
      icon: Sliders,
      phase: 'Phase 2',
      requiresImage: true,
    },
    {
      id: 'reconstruction' as ActiveTab,
      label: 'Low-Rank Reconstruction',
      sublabel: 'A_k = U_k Σ_k V_kᵀ & PSNR',
      icon: Sparkles,
      phase: 'Phase 2',
      requiresImage: true,
    },
  ];

  return (
    <aside className="w-72 shrink-0 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex flex-col p-4 transition-colors">
      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3 px-2">
        Analysis Modules
      </div>
      <nav className="space-y-1.5 flex-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          const isDisabled = item.requiresImage && !hasImage;

          return (
            <button
              key={item.id}
              disabled={isDisabled}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-start text-left px-3 py-2.5 rounded-lg text-sm transition-all ${isActive
                  ? 'bg-sky-50 dark:bg-sky-950/60 text-sky-800 dark:text-sky-300 font-medium border border-sky-200 dark:border-sky-800/80 shadow-sm'
                  : isDisabled
                    ? 'opacity-40 cursor-not-allowed text-slate-400 dark:text-slate-600'
                    : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/60'
                }`}
            >
              <Icon className={`w-4 h-4 mt-0.5 mr-2.5 shrink-0 ${isActive ? 'text-sky-600 dark:text-sky-400' : 'text-slate-400'}`} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between">
                  <span className="truncate">{item.label}</span>
                  <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded ${item.phase === 'Phase 1'
                      ? 'bg-slate-100 dark:bg-slate-800 text-slate-500'
                      : 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800/50'
                    }`}>
                    {item.phase}
                  </span>
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400 font-mono truncate mt-0.5">
                  {item.sublabel}
                </div>
              </div>
            </button>
          );
        })}
      </nav>

      <div className="pt-4 border-t border-slate-200 dark:border-slate-800 text-xs text-slate-500 dark:text-slate-400 space-y-1">
        <div className="flex justify-between">
          <span>Engine:</span>
          <span className="font-mono text-slate-700 dark:text-slate-300">NumPy + SciPy</span>
        </div>
        <div className="flex justify-between">
          <span>Floating Point:</span>
          <span className="font-mono text-slate-700 dark:text-slate-300">IEEE 754 (Float64)</span>
        </div>
        <div className="flex justify-between">
          <span>Precision:</span>
          <span className="font-mono text-slate-700 dark:text-slate-300">BT.601 / TSVD</span>
        </div>
      </div>
    </aside>
  );
};
