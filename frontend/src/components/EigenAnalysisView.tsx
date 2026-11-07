import React, { useState, useEffect } from 'react';
import { Compass } from 'lucide-react';
import type { ImageMetadata, EigenAnalysisData } from '../types';
import { getEigenAnalysis } from '../services/api';
import { ScientificPlot } from './ScientificPlot';

interface EigenAnalysisViewProps {
  metadata: ImageMetadata;
  darkMode: boolean;
}

export const EigenAnalysisView: React.FC<EigenAnalysisViewProps> = ({ metadata, darkMode }) => {
  const [matrixType, setMatrixType] = useState<'gram_row' | 'gram_col' | 'covariance' | 'patch'>('gram_row');
  const [patchSize, setPatchSize] = useState<number>(8);
  const [eigenData, setEigenData] = useState<EigenAnalysisData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchEigen = (type: typeof matrixType, size: number) => {
    setLoading(true);
    getEigenAnalysis(metadata.image_id, type, undefined, undefined, size)
      .then((data) => {
        setEigenData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchEigen(matrixType, patchSize);
  }, [metadata.image_id, matrixType, patchSize]);

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
          Spectral Decomposition: Eigenvalue & Eigenvector Analysis
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Characteristic equation <span className="font-mono">Av = λv</span>, invariant subspace directions, and Gram/Covariance spectra.
        </p>
      </div>

      <div className="rounded-xl border border-sky-300 dark:border-sky-800/80 bg-sky-50/70 dark:bg-sky-950/20 p-5 space-y-2.5">
        <div className="flex items-center space-x-2 text-sky-900 dark:text-sky-300 text-sm font-semibold">
          <Compass className="w-5 h-5 text-sky-600 dark:text-sky-400 shrink-0" />
          <span>Rectangular Matrices, Gram Formulations, and PCA Distinction</span>
        </div>
        <p className="text-xs text-sky-950/90 dark:text-sky-200/90 leading-relaxed font-sans">
          {eigenData?.pca_distinction ||
            'Conventional eigenvalue decomposition Av = λv is mathematically defined strictly for square endomorphisms (M ∈ ℝ^(n×n)). Because image matrices are generally rectangular (H ≠ W), a square proxy (Gram matrix AAᵀ, AᵀA, covariance, or a localized square patch) is analyzed.'}
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <div className="space-y-1">
          <span className="text-xs font-semibold text-slate-900 dark:text-white">
            Select Square Operator for Eigen Analysis:
          </span>
          <div className="flex flex-wrap gap-2 pt-1">
            <button
              onClick={() => setMatrixType('gram_row')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                matrixType === 'gram_row'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200'
              }`}
            >
              Row Gram Matrix (AAᵀ)
            </button>
            <button
              onClick={() => setMatrixType('gram_col')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                matrixType === 'gram_col'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200'
              }`}
            >
              Col Gram Matrix (AᵀA)
            </button>
            <button
              onClick={() => setMatrixType('covariance')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                matrixType === 'covariance'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200'
              }`}
            >
              Centered Covariance Matrix
            </button>
            <button
              onClick={() => setMatrixType('patch')}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                matrixType === 'patch'
                  ? 'bg-sky-600 text-white font-semibold'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200'
              }`}
            >
              Local Image Patch (Direct Pv = λv)
            </button>
          </div>
        </div>

        {matrixType === 'patch' && (
          <div className="flex items-center space-x-2 text-xs font-mono">
            <span className="text-slate-500">Patch Size:</span>
            {[4, 8, 16].map((s) => (
              <button
                key={s}
                onClick={() => setPatchSize(s)}
                className={`px-2 py-1 rounded text-xs ${
                  patchSize === s
                    ? 'bg-slate-800 text-white dark:bg-white dark:text-slate-900 font-bold'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300'
                }`}
              >
                {s}×{s}
              </button>
            ))}
          </div>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center p-12 text-slate-400 font-mono text-sm">
          Computing characteristic roots det(A - λI) = 0 and invariant eigenvectors...
        </div>
      ) : eigenData ? (
        <>
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-2 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-900 dark:text-white uppercase tracking-wider">
                Operator Description & Dimensions
              </span>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                {eigenData.matrix_dimensions[0]} × {eigenData.matrix_dimensions[1]}
              </span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-300 font-sans leading-relaxed">
              {eigenData.matrix_description}
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-4 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2">
              <span className="text-xs font-semibold text-slate-900 dark:text-white">
                Eigenvalue Spectrum |λᵢ| (Sorted by Magnitude)
              </span>
              {eigenData.has_complex && (
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-amber-100 dark:bg-amber-950 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-800">
                  Complex Eigenvalues Present (λ = α ± iβ)
                </span>
              )}
            </div>

            <ScientificPlot
              data={[
                {
                  x: eigenData.eigenvalues_real.map((_, i) => i + 1),
                  y: eigenData.eigenvalues_real.map((r, i) =>
                    Math.sqrt(r * r + (eigenData.eigenvalues_imag[i] || 0) ** 2)
                  ),
                  type: 'scatter',
                  mode: 'lines+markers',
                  marker: { color: '#0284c7', size: 6 },
                  line: { color: '#0284c7', width: 2 },
                  name: '|λᵢ| Magnitude',
                },
              ]}
              layout={{
                autosize: true,
                height: 320,
                margin: { l: 50, r: 25, t: 20, b: 45 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: {
                  title: { text: 'Eigen-index i', font: { size: 11, color: darkMode ? '#94a3b8' : '#64748b' } },
                  tickfont: { color: darkMode ? '#94a3b8' : '#64748b' },
                  gridcolor: darkMode ? '#1e293b' : '#f1f5f9',
                },
                yaxis: {
                  title: { text: 'Eigenvalue Magnitude |λ|', font: { size: 11, color: darkMode ? '#94a3b8' : '#64748b' } },
                  tickfont: { color: darkMode ? '#94a3b8' : '#64748b' },
                  gridcolor: darkMode ? '#1e293b' : '#f1f5f9',
                },
              }}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-3 shadow-sm">
              <span className="text-xs font-semibold text-slate-900 dark:text-white">
                Submatrix Numerical Representation (Top {eigenData.sample_matrix_preview.length}×{eigenData.sample_matrix_preview.length})
              </span>
              <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-lg p-2 bg-slate-50 dark:bg-slate-950 font-mono text-[11px]">
                <table className="w-full text-center">
                  <tbody>
                    {eigenData.sample_matrix_preview.map((row, rIdx) => (
                      <tr key={rIdx}>
                        {row.map((val, cIdx) => (
                          <td key={cIdx} className="p-1 text-slate-700 dark:text-slate-300">
                            {val.toFixed(1)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-3 shadow-sm">
              <span className="text-xs font-semibold text-slate-900 dark:text-white">
                Top Calculated Eigenvalues (Roots of det(A - λI)=0)
              </span>
              <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                {eigenData.eigenvalues_real.slice(0, 10).map((r, i) => {
                  const im = eigenData.eigenvalues_imag[i];
                  const hasIm = Math.abs(im) > 1e-4;
                  return (
                    <div
                      key={i}
                      className="flex items-center justify-between p-2 rounded bg-slate-50 dark:bg-slate-800/60 font-mono text-xs border border-slate-200/50 dark:border-slate-700/50"
                    >
                      <span className="text-slate-500 font-semibold">λ_{i + 1}</span>
                      <span className="text-slate-900 dark:text-white">
                        {r.toFixed(4)} {hasIm ? `${im >= 0 ? '+' : '-'} ${Math.abs(im).toFixed(4)}i` : ''}
                      </span>
                      <span className="text-slate-400 text-[10px]">
                        energy: {eigenData.eigenvalues_energy_distribution[i]}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
