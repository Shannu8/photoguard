import React, { useState, useEffect, useRef } from 'react';
import { Target, Calculator } from 'lucide-react';
import type { ImageMetadata, PixelInspection } from '../types';
import { inspectPixel } from '../services/api';

interface PixelInspectorViewProps {
  metadata: ImageMetadata;
}

export const PixelInspectorView: React.FC<PixelInspectorViewProps> = ({ metadata }) => {
  const [selectedCoord, setSelectedCoord] = useState<{ x: number; y: number }>({
    x: Math.floor(metadata.width / 2),
    y: Math.floor(metadata.height / 2),
  });
  const [radius, setRadius] = useState<number>(3); // (2*3+1) = 7x7 matrix
  const [inspection, setInspection] = useState<PixelInspection | null>(null);
  const [activeChannelView, setActiveChannelView] = useState<'rgb' | 'r' | 'g' | 'b' | 'gray'>('rgb');
  const imgRef = useRef<HTMLImageElement>(null);

  const fetchPixelData = async (x: number, y: number, r: number) => {
    try {
      const data = await inspectPixel(metadata.image_id, x, y, r);
      setInspection(data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchPixelData(selectedCoord.x, selectedCoord.y, radius);
  }, [metadata.image_id, selectedCoord.x, selectedCoord.y, radius]);

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const scaleX = metadata.width / rect.width;
    const scaleY = metadata.height / rect.height;

    const clickX = Math.floor((e.clientX - rect.left) * scaleX);
    const clickY = Math.floor((e.clientY - rect.top) * scaleY);

    const clampedX = Math.max(0, Math.min(metadata.width - 1, clickX));
    const clampedY = Math.max(0, Math.min(metadata.height - 1, clickY));

    setSelectedCoord({ x: clampedX, y: clampedY });
  };

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-white">
          RGB Matrix Representation & Pixel Inspector
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Inspect individual matrix elements <span className="font-mono">I[y, x] = [R, G, B]</span> and examine local submatrix neighborhoods.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-sm font-semibold text-slate-900 dark:text-white">
              <Target className="w-4 h-4 text-sky-500" />
              <span>Interactive Coordinate Selector</span>
            </div>
            <span className="text-xs text-slate-400 font-mono">
              Click anywhere on image
            </span>
          </div>

          <div className="relative border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center p-1">
            <img
              ref={imgRef}
              src={metadata.rgb_data_url}
              alt="Inspectable matrix"
              onClick={handleImageClick}
              className="max-h-80 w-auto object-contain cursor-crosshair select-none"
            />
          </div>

          <div className="grid grid-cols-2 gap-3 pt-2">
            <div>
              <label className="text-xs font-mono text-slate-500 dark:text-slate-400">
                Column X (0 .. {metadata.width - 1}):
              </label>
              <input
                type="number"
                min={0}
                max={metadata.width - 1}
                value={selectedCoord.x}
                onChange={(e) =>
                  setSelectedCoord((prev) => ({
                    ...prev,
                    x: Math.max(0, Math.min(metadata.width - 1, parseInt(e.target.value) || 0)),
                  }))
                }
                className="w-full mt-1 px-3 py-1.5 text-xs font-mono rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white"
              />
            </div>
            <div>
              <label className="text-xs font-mono text-slate-500 dark:text-slate-400">
                Row Y (0 .. {metadata.height - 1}):
              </label>
              <input
                type="number"
                min={0}
                max={metadata.height - 1}
                value={selectedCoord.y}
                onChange={(e) =>
                  setSelectedCoord((prev) => ({
                    ...prev,
                    y: Math.max(0, Math.min(metadata.height - 1, parseInt(e.target.value) || 0)),
                  }))
                }
                className="w-full mt-1 px-3 py-1.5 text-xs font-mono rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white"
              />
            </div>
          </div>

          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 pt-1">
            <span>Neighborhood Window Size:</span>
            <div className="flex space-x-1.5">
              {[1, 2, 3, 4].map((r) => (
                <button
                  key={r}
                  onClick={() => setRadius(r)}
                  className={`px-2 py-0.5 rounded text-xs font-mono ${
                    radius === r
                      ? 'bg-sky-600 text-white font-medium'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                  }`}
                >
                  {2 * r + 1}×{2 * r + 1}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="lg:col-span-7 space-y-6">
          {inspection && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-5 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-sm text-slate-900 dark:text-white">
                    Pixel Element at (x={inspection.x}, y={inspection.y})
                  </span>
                  <div
                    className="w-5 h-5 rounded border border-slate-300 dark:border-slate-700 shadow-inner"
                    style={{ backgroundColor: `rgb(${inspection.r}, ${inspection.g}, ${inspection.b})` }}
                    title={`RGB(${inspection.r}, ${inspection.g}, ${inspection.b})`}
                  />
                </div>
                <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                  Matrix Entry: I[{inspection.y}, {inspection.x}]
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-lg bg-red-50/70 dark:bg-red-950/30 border border-red-200 dark:border-red-900/40">
                  <div className="text-[11px] font-semibold text-red-700 dark:text-red-400">Red Channel (R)</div>
                  <div className="font-mono text-lg font-bold text-red-900 dark:text-red-300 mt-0.5">
                    {inspection.r}
                  </div>
                  <div className="text-[10px] text-red-600/80 font-mono">
                    norm: {(inspection.r / 255).toFixed(3)}
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/40">
                  <div className="text-[11px] font-semibold text-emerald-700 dark:text-emerald-400">Green Channel (G)</div>
                  <div className="font-mono text-lg font-bold text-emerald-900 dark:text-emerald-300 mt-0.5">
                    {inspection.g}
                  </div>
                  <div className="text-[10px] text-emerald-600/80 font-mono">
                    norm: {(inspection.g / 255).toFixed(3)}
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-blue-50/70 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900/40">
                  <div className="text-[11px] font-semibold text-blue-700 dark:text-blue-400">Blue Channel (B)</div>
                  <div className="font-mono text-lg font-bold text-blue-900 dark:text-blue-300 mt-0.5">
                    {inspection.b}
                  </div>
                  <div className="text-[10px] text-blue-600/80 font-mono">
                    norm: {(inspection.b / 255).toFixed(3)}
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                  <div className="text-[11px] font-semibold text-slate-700 dark:text-slate-300">Grayscale (Y)</div>
                  <div className="font-mono text-lg font-bold text-slate-900 dark:text-white mt-0.5">
                    {inspection.grayscale_uint8}
                  </div>
                  <div className="text-[10px] text-slate-500 font-mono">
                    exact: {inspection.grayscale_exact}
                  </div>
                </div>
              </div>

              <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 space-y-1.5">
                <div className="flex items-center space-x-1.5 text-xs font-semibold text-slate-700 dark:text-slate-300">
                  <Calculator className="w-3.5 h-3.5 text-sky-500" />
                  <span>Mathematical Pixel Calculation (ITU-R BT.601):</span>
                </div>
                <pre className="font-mono text-xs text-sky-800 dark:text-sky-300 bg-white dark:bg-slate-900 p-2.5 rounded border border-slate-200/60 dark:border-slate-800 whitespace-pre overflow-x-auto">
                  {inspection.calculation_steps}
                </pre>
              </div>

              <div className="space-y-3 pt-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                    Neighborhood Submatrix Window ({2 * radius + 1}×{2 * radius + 1})
                  </span>
                  <div className="flex space-x-1">
                    {(['rgb', 'r', 'g', 'b', 'gray'] as const).map((view) => (
                      <button
                        key={view}
                        onClick={() => setActiveChannelView(view)}
                        className={`px-2 py-0.5 text-xs font-mono rounded ${
                          activeChannelView === view
                            ? 'bg-slate-800 text-white dark:bg-white dark:text-slate-900 font-medium'
                            : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200'
                        }`}
                      >
                        {view.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-lg p-2 bg-slate-50/50 dark:bg-slate-950">
                  <table className="w-full text-center border-collapse">
                    <thead>
                      <tr>
                        <th className="p-1 text-[10px] font-mono text-slate-400">Y \ X</th>
                        {inspection.submatrix_r[0]?.map((_, colIdx) => (
                          <th key={colIdx} className="p-1 text-[10px] font-mono text-slate-400">
                            {inspection.submatrix_bounds.col_start + colIdx}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {inspection.submatrix_r.map((row, rowIdx) => {
                        const actualY = inspection.submatrix_bounds.row_start + rowIdx;
                        return (
                          <tr key={rowIdx}>
                            <td className="p-1 text-[10px] font-mono text-slate-400 font-medium">
                              {actualY}
                            </td>
                            {row.map((_, colIdx) => {
                              const actualX = inspection.submatrix_bounds.col_start + colIdx;
                              const isCenter = actualX === inspection.x && actualY === inspection.y;

                              const rVal = inspection.submatrix_r[rowIdx][colIdx];
                              const gVal = inspection.submatrix_g[rowIdx][colIdx];
                              const bVal = inspection.submatrix_b[rowIdx][colIdx];
                              const grayVal = inspection.submatrix_grayscale[rowIdx][colIdx];

                              return (
                                <td
                                  key={colIdx}
                                  className={`p-1.5 text-xs font-mono transition-colors ${
                                    isCenter
                                      ? 'ring-2 ring-sky-500 rounded font-bold bg-sky-100/60 dark:bg-sky-950/80 text-sky-900 dark:text-sky-200'
                                      : 'text-slate-700 dark:text-slate-300'
                                  }`}
                                >
                                  {activeChannelView === 'rgb' && (
                                    <div className="flex items-center justify-center space-x-1">
                                      <span
                                        className="w-2.5 h-2.5 rounded-full inline-block shrink-0"
                                        style={{ backgroundColor: `rgb(${rVal}, ${gVal}, ${bVal})` }}
                                      />
                                      <span className="text-[10px] font-mono">
                                        [{rVal},{gVal},{bVal}]
                                      </span>
                                    </div>
                                  )}
                                  {activeChannelView === 'r' && <span className="text-red-600 dark:text-red-400">{rVal}</span>}
                                  {activeChannelView === 'g' && <span className="text-emerald-600 dark:text-emerald-400">{gVal}</span>}
                                  {activeChannelView === 'b' && <span className="text-blue-600 dark:text-blue-400">{bVal}</span>}
                                  {activeChannelView === 'gray' && <span>{grayVal}</span>}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
