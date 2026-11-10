import React, { useState, useEffect, useRef } from 'react';
import {
  RotateCw,
  Maximize2,
  FlipHorizontal,
  Move,
  Layers,
  Download,
  RotateCcw,
  ArrowUp,
  ArrowDown,
  Trash2,
  Plus,
  Target,
  FileSpreadsheet,
  Database,
  Info,
  CheckCircle2,
  Sliders,
  Scissors,
} from 'lucide-react';
import type {
  ImageMetadata,
  TransformationStep,
  TransformationResponse,
  TensorMetadataResponse,
} from '../types';
import {
  applyTransformations,
  getTensorMetadata,
  getTensorDownloadUrl,
} from '../services/api';

interface TransformationsViewProps {
  metadata: ImageMetadata;
  darkMode?: boolean;
}

export const TransformationsView: React.FC<TransformationsViewProps> = ({
  metadata,
}) => {
  // Transformation sequence
  const [operations, setOperations] = useState<TransformationStep[]>([
    { id: '1', type: 'rotate', params: { angle: 45 } },
  ]);

  // Selected tool to add/edit
  const [selectedOpType, setSelectedOpType] = useState<TransformationStep['type']>('rotate');
  const [rotateAngle, setRotateAngle] = useState<number>(45);
  const [scaleX, setScaleX] = useState<number>(1.2);
  const [scaleY, setScaleY] = useState<number>(1.2);
  const [uniformScale, setUniformScale] = useState<boolean>(true);
  const [reflectAxis, setReflectAxis] = useState<'horizontal' | 'vertical' | 'both'>('horizontal');
  const [shearKx, setShearKx] = useState<number>(0.2);
  const [shearKy, setShearKy] = useState<number>(0.0);
  const [transX, setTransX] = useState<number>(0);
  const [transY, setTransY] = useState<number>(0);

  // Settings
  const [interpolation, setInterpolation] = useState<'bilinear' | 'nearest'>('bilinear');
  const [canvasMode, setCanvasMode] = useState<'fit' | 'same'>('fit');
  const [selectedPoint, setSelectedPoint] = useState<[number, number]>([
    Math.floor(metadata.width / 2),
    Math.floor(metadata.height / 2),
  ]);

  // Results & Loading
  const [loading, setLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<TransformationResponse | null>(null);
  const [tensorMeta, setTensorMeta] = useState<TensorMetadataResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const origImgRef = useRef<HTMLImageElement>(null);

  // Fetch transformation
  const handleExecute = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await applyTransformations(
        metadata.image_id,
        operations,
        interpolation,
        canvasMode,
        selectedPoint
      );
      setResponse(res);

      // Refresh tensor metadata
      const tMeta = await getTensorMetadata(metadata.image_id);
      setTensorMeta(tMeta);
    } catch (err: any) {
      setErrorMsg(err.message || 'Transformation failed.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleExecute();
  }, [metadata.image_id, operations, interpolation, canvasMode, selectedPoint]);

  // Point click on original image
  const handleOriginalImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!origImgRef.current) return;
    const rect = origImgRef.current.getBoundingClientRect();
    const scaleX = metadata.width / rect.width;
    const scaleY = metadata.height / rect.height;

    const clickX = Math.round((e.clientX - rect.left) * scaleX);
    const clickY = Math.round((e.clientY - rect.top) * scaleY);

    const clampedX = Math.max(0, Math.min(metadata.width - 1, clickX));
    const clampedY = Math.max(0, Math.min(metadata.height - 1, clickY));

    setSelectedPoint([clampedX, clampedY]);
  };

  // Add step to composition pipeline
  const handleAddStep = () => {
    let params: Record<string, any> = {};
    if (selectedOpType === 'rotate') params = { angle: rotateAngle };
    else if (selectedOpType === 'scale') params = { sx: scaleX, sy: scaleY };
    else if (selectedOpType === 'reflect') params = { axis: reflectAxis };
    else if (selectedOpType === 'shear') params = { kx: shearKx, ky: shearKy };
    else if (selectedOpType === 'translate') params = { tx: transX, ty: transY };

    const newStep: TransformationStep = {
      id: Date.now().toString(),
      type: selectedOpType,
      params,
    };
    setOperations((prev) => [...prev, newStep]);
  };

  const handleRemoveStep = (index: number) => {
    setOperations((prev) => prev.filter((_, i) => i !== index));
  };

  const handleMoveStep = (index: number, direction: 'up' | 'down') => {
    setOperations((prev) => {
      const copy = [...prev];
      const targetIndex = direction === 'up' ? index - 1 : index + 1;
      if (targetIndex < 0 || targetIndex >= copy.length) return prev;
      const temp = copy[index];
      copy[index] = copy[targetIndex];
      copy[targetIndex] = temp;
      return copy;
    });
  };

  const handleReset = () => {
    setOperations([]);
    setSelectedPoint([Math.floor(metadata.width / 2), Math.floor(metadata.height / 2)]);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="space-y-8 max-w-7xl pb-16">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2">
          <span className="text-xs font-mono font-semibold uppercase px-2.5 py-0.5 rounded-full bg-sky-100 dark:bg-sky-950/80 text-sky-800 dark:text-sky-300 border border-sky-300 dark:border-sky-800">
            Phase 3
          </span>
          <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">
            Mathematical Image Transformations & Tensor Export
          </h2>
        </div>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-4xl">
          Apply 2D affine transformations using homogeneous coordinates{' '}
          <span className="font-mono text-sky-600 dark:text-sky-400">p' = M · p</span>. Inspect
          individual operator matrices, demonstrate non-commutative composition{' '}
          <span className="font-mono text-sky-600 dark:text-sky-400">AB ≠ BA</span>, and download the
          exact uncompressed multidimensional image tensors.
        </p>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl border border-rose-300 dark:border-rose-900 bg-rose-50 dark:bg-rose-950/40 text-rose-800 dark:text-rose-300 text-sm flex items-center space-x-2">
          <Info className="w-5 h-5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Grid: Controls & Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Transformation Builder & Pipeline */}
        <div className="lg:col-span-5 space-y-6">
          {/* 1. Operation Configurator */}
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
              <div className="flex items-center space-x-2 text-sm font-semibold text-slate-900 dark:text-white">
                <Sliders className="w-4 h-4 text-sky-500" />
                <span>Transformation Operator</span>
              </div>
              <span className="text-xs font-mono text-slate-400">Centered Affine</span>
            </div>

            {/* Operator Selector Tabs */}
            <div className="grid grid-cols-5 gap-1 p-1 bg-slate-100 dark:bg-slate-800/80 rounded-lg text-xs font-medium">
              {(
                [
                  { id: 'rotate', label: 'Rotate', icon: RotateCw },
                  { id: 'scale', label: 'Scale', icon: Maximize2 },
                  { id: 'reflect', label: 'Reflect', icon: FlipHorizontal },
                  { id: 'shear', label: 'Shear', icon: Scissors },
                  { id: 'translate', label: 'Shift', icon: Move },
                ] as const
              ).map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setSelectedOpType(tab.id)}
                    className={`flex flex-col items-center py-2 px-1 rounded-md transition-all ${
                      selectedOpType === tab.id
                        ? 'bg-white dark:bg-slate-900 text-sky-600 dark:text-sky-400 shadow-xs font-semibold'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5 mb-1" />
                    <span className="text-[11px]">{tab.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Parameter Panels */}
            {selectedOpType === 'rotate' && (
              <div className="space-y-3 pt-1">
                <div className="flex justify-between items-center text-xs font-mono text-slate-600 dark:text-slate-300">
                  <span>Angle (θ):</span>
                  <span className="font-bold text-sky-600 dark:text-sky-400">{rotateAngle}°</span>
                </div>
                <input
                  type="range"
                  min={-180}
                  max={180}
                  step={1}
                  value={rotateAngle}
                  onChange={(e) => setRotateAngle(parseFloat(e.target.value))}
                  className="w-full accent-sky-500 cursor-pointer"
                />
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {[-180, -90, -45, 45, 90, 180, 270].map((deg) => (
                    <button
                      key={deg}
                      onClick={() => setRotateAngle(deg)}
                      className={`px-2 py-0.5 text-xs font-mono rounded border ${
                        rotateAngle === deg
                          ? 'bg-sky-50 dark:bg-sky-950 text-sky-700 dark:text-sky-300 border-sky-300 dark:border-sky-700 font-bold'
                          : 'border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800'
                      }`}
                    >
                      {deg > 0 ? `+${deg}°` : `${deg}°`}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {selectedOpType === 'scale' && (
              <div className="space-y-3 pt-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-mono text-slate-600 dark:text-slate-300 flex items-center space-x-1.5">
                    <input
                      type="checkbox"
                      checked={uniformScale}
                      onChange={(e) => setUniformScale(e.target.checked)}
                      className="rounded accent-sky-500"
                    />
                    <span>Uniform Scaling (sx = sy)</span>
                  </label>
                  <span className="text-xs font-mono font-bold text-sky-600 dark:text-sky-400">
                    {scaleX.toFixed(2)}x {uniformScale ? '' : `, ${scaleY.toFixed(2)}x`}
                  </span>
                </div>
                <div className="space-y-2">
                  <div>
                    <div className="text-[11px] font-mono text-slate-400">X Scale Factor (sx):</div>
                    <input
                      type="range"
                      min={0.2}
                      max={3.0}
                      step={0.05}
                      value={scaleX}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        setScaleX(val);
                        if (uniformScale) setScaleY(val);
                      }}
                      className="w-full accent-sky-500 cursor-pointer"
                    />
                  </div>
                  {!uniformScale && (
                    <div>
                      <div className="text-[11px] font-mono text-slate-400">Y Scale Factor (sy):</div>
                      <input
                        type="range"
                        min={0.2}
                        max={3.0}
                        step={0.05}
                        value={scaleY}
                        onChange={(e) => setScaleY(parseFloat(e.target.value))}
                        className="w-full accent-sky-500 cursor-pointer"
                      />
                    </div>
                  )}
                </div>
                <div className="flex gap-1.5 pt-1">
                  {[0.5, 0.75, 1.0, 1.5, 2.0].map((f) => (
                    <button
                      key={f}
                      onClick={() => {
                        setScaleX(f);
                        setScaleY(f);
                      }}
                      className="px-2 py-0.5 text-xs font-mono rounded border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"
                    >
                      {f}x
                    </button>
                  ))}
                </div>
              </div>
            )}

            {selectedOpType === 'reflect' && (
              <div className="space-y-3 pt-1">
                <div className="text-xs font-mono text-slate-500 dark:text-slate-400">
                  Select reflection axis centered across the image:
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { id: 'horizontal', label: 'Left ↔ Right' },
                    { id: 'vertical', label: 'Top ↕ Bottom' },
                    { id: 'both', label: 'Both (Invert)' },
                  ].map((ax) => (
                    <button
                      key={ax.id}
                      onClick={() => setReflectAxis(ax.id as any)}
                      className={`py-2 px-2 text-xs font-mono rounded-lg border text-center transition-all ${
                        reflectAxis === ax.id
                          ? 'bg-sky-50 dark:bg-sky-950 text-sky-700 dark:text-sky-300 border-sky-400 dark:border-sky-700 font-bold'
                          : 'border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-50'
                      }`}
                    >
                      {ax.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {selectedOpType === 'shear' && (
              <div className="space-y-3 pt-1">
                <div>
                  <div className="flex justify-between text-xs font-mono text-slate-600 dark:text-slate-300">
                    <span>X-Axis Shear (kx):</span>
                    <span className="font-bold text-sky-600 dark:text-sky-400">{shearKx.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={-1.0}
                    max={1.0}
                    step={0.05}
                    value={shearKx}
                    onChange={(e) => setShearKx(parseFloat(e.target.value))}
                    className="w-full accent-sky-500 cursor-pointer"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono text-slate-600 dark:text-slate-300">
                    <span>Y-Axis Shear (ky):</span>
                    <span className="font-bold text-sky-600 dark:text-sky-400">{shearKy.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={-1.0}
                    max={1.0}
                    step={0.05}
                    value={shearKy}
                    onChange={(e) => setShearKy(parseFloat(e.target.value))}
                    className="w-full accent-sky-500 cursor-pointer"
                  />
                </div>
              </div>
            )}

            {selectedOpType === 'translate' && (
              <div className="space-y-3 pt-1">
                <div>
                  <div className="flex justify-between text-xs font-mono text-slate-600 dark:text-slate-300">
                    <span>X Displacement (tx):</span>
                    <span className="font-bold text-sky-600 dark:text-sky-400">{transX} px</span>
                  </div>
                  <input
                    type="range"
                    min={-Math.floor(metadata.width / 2)}
                    max={Math.floor(metadata.width / 2)}
                    step={1}
                    value={transX}
                    onChange={(e) => setTransX(parseInt(e.target.value))}
                    className="w-full accent-sky-500 cursor-pointer"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono text-slate-600 dark:text-slate-300">
                    <span>Y Displacement (ty):</span>
                    <span className="font-bold text-sky-600 dark:text-sky-400">{transY} px</span>
                  </div>
                  <input
                    type="range"
                    min={-Math.floor(metadata.height / 2)}
                    max={Math.floor(metadata.height / 2)}
                    step={1}
                    value={transY}
                    onChange={(e) => setTransY(parseInt(e.target.value))}
                    className="w-full accent-sky-500 cursor-pointer"
                  />
                </div>
              </div>
            )}

            {/* Add to Pipeline Button */}
            <button
              onClick={handleAddStep}
              className="w-full py-2 px-3 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold flex items-center justify-center space-x-1.5 transition-colors shadow-xs"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Add Step to Transformation Pipeline</span>
            </button>
          </div>

          {/* 2. Pipeline Manager */}
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
              <div className="flex items-center space-x-2 text-sm font-semibold text-slate-900 dark:text-white">
                <Layers className="w-4 h-4 text-sky-500" />
                <span>Transformation Pipeline ({operations.length} steps)</span>
              </div>
              <button
                onClick={handleReset}
                className="text-xs text-rose-600 dark:text-rose-400 hover:underline flex items-center space-x-1"
                title="Reset to identity matrix"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Reset</span>
              </button>
            </div>

            {operations.length === 0 ? (
              <div className="py-6 text-center text-xs text-slate-400 font-mono">
                Pipeline is empty (Identity Matrix I₃). Click "Add Step" above to compose operations.
              </div>
            ) : (
              <div className="space-y-2">
                {operations.map((op, idx) => (
                  <div
                    key={op.id || idx}
                    className="flex items-center justify-between p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 text-xs font-mono"
                  >
                    <div className="flex items-center space-x-2">
                      <span className="w-5 h-5 rounded-full bg-sky-100 dark:bg-sky-900 text-sky-700 dark:text-sky-300 flex items-center justify-center font-bold text-[10px]">
                        {idx + 1}
                      </span>
                      <div>
                        <span className="font-bold uppercase text-slate-800 dark:text-slate-200">
                          {op.type}
                        </span>
                        <span className="text-slate-500 dark:text-slate-400 ml-1.5 text-[11px]">
                          {op.type === 'rotate' && `(${op.params.angle}°)`}
                          {op.type === 'scale' && `(sx=${op.params.sx}, sy=${op.params.sy})`}
                          {op.type === 'reflect' && `(${op.params.axis})`}
                          {op.type === 'shear' && `(kx=${op.params.kx}, ky=${op.params.ky})`}
                          {op.type === 'translate' && `(Δx=${op.params.tx}, Δy=${op.params.ty})`}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center space-x-1">
                      <button
                        disabled={idx === 0}
                        onClick={() => handleMoveStep(idx, 'up')}
                        className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-30 text-slate-600 dark:text-slate-300"
                        title="Move step up (alter composition order)"
                      >
                        <ArrowUp className="w-3.5 h-3.5" />
                      </button>
                      <button
                        disabled={idx === operations.length - 1}
                        onClick={() => handleMoveStep(idx, 'down')}
                        className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-30 text-slate-600 dark:text-slate-300"
                        title="Move step down (alter composition order)"
                      >
                        <ArrowDown className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleRemoveStep(idx)}
                        className="p-1 rounded hover:bg-rose-100 dark:hover:bg-rose-950 text-rose-600 dark:text-rose-400"
                        title="Remove step"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Settings Row */}
            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-100 dark:border-slate-800 text-xs">
              <div>
                <label className="text-[11px] font-mono text-slate-500 dark:text-slate-400 block mb-1">
                  Interpolation Kernel:
                </label>
                <select
                  value={interpolation}
                  onChange={(e) => setInterpolation(e.target.value as any)}
                  className="w-full px-2.5 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-xs"
                >
                  <option value="bilinear">Bilinear (Smooth)</option>
                  <option value="nearest">Nearest Neighbor (Discrete)</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-mono text-slate-500 dark:text-slate-400 block mb-1">
                  Canvas Boundary:
                </label>
                <select
                  value={canvasMode}
                  onChange={(e) => setCanvasMode(e.target.value as any)}
                  className="w-full px-2.5 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white font-mono text-xs"
                >
                  <option value="fit">Fit Canvas (No Crop)</option>
                  <option value="same">Preserve Canvas ({metadata.width}×{metadata.height})</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Visual Comparison & Math */}
        <div className="lg:col-span-7 space-y-6">
          {/* Side-by-Side Images */}
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
              <span className="font-semibold text-sm text-slate-900 dark:text-white flex items-center space-x-2">
                <span>Side-by-Side Visual Comparison</span>
              </span>
              <span className="text-xs font-mono text-slate-400">
                Click original image to map coordinate (x, y)
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Original Image */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-mono text-slate-500 dark:text-slate-400">
                  <span className="font-bold text-slate-800 dark:text-slate-200">Original Matrix</span>
                  <span>{metadata.width} × {metadata.height}</span>
                </div>
                <div className="relative border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[240px]">
                  <img
                    ref={origImgRef}
                    src={metadata.rgb_data_url}
                    alt="Original"
                    onClick={handleOriginalImageClick}
                    className="max-h-60 w-auto object-contain cursor-crosshair select-none"
                  />
                  {selectedPoint && (
                    <div
                      className="absolute w-3 h-3 rounded-full border-2 border-white bg-sky-500 shadow-md transform -translate-x-1/2 -translate-y-1/2 pointer-events-none"
                      style={{
                        left: `${(selectedPoint[0] / metadata.width) * 100}%`,
                        top: `${(selectedPoint[1] / metadata.height) * 100}%`,
                      }}
                    />
                  )}
                </div>
                <div className="text-[11px] font-mono text-slate-500 dark:text-slate-400 flex justify-between">
                  <span>Selected Point p:</span>
                  <span className="font-bold text-sky-600 dark:text-sky-400">
                    ({selectedPoint[0]}, {selectedPoint[1]})
                  </span>
                </div>
              </div>

              {/* Transformed Image */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-mono text-slate-500 dark:text-slate-400">
                  <span className="font-bold text-slate-800 dark:text-slate-200">Transformed Matrix</span>
                  <span>
                    {response ? `${response.transformed_width} × ${response.transformed_height}` : '...'}
                  </span>
                </div>
                <div className="relative border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[240px]">
                  {loading ? (
                    <div className="flex flex-col items-center space-y-2 text-xs font-mono text-slate-400">
                      <div className="w-6 h-6 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
                      <span>Computing warp...</span>
                    </div>
                  ) : response ? (
                    <>
                      <img
                        src={response.transformed_rgb_data_url}
                        alt="Transformed"
                        className="max-h-60 w-auto object-contain select-none"
                      />
                      {response.transformed_point && (
                        <div
                          className="absolute w-3 h-3 rounded-full border-2 border-white bg-emerald-500 shadow-md transform -translate-x-1/2 -translate-y-1/2 pointer-events-none"
                          style={{
                            left: `${(response.transformed_point[0] / response.transformed_width) * 100}%`,
                            top: `${(response.transformed_point[1] / response.transformed_height) * 100}%`,
                          }}
                        />
                      )}
                    </>
                  ) : null}
                </div>
                <div className="text-[11px] font-mono text-slate-500 dark:text-slate-400 flex justify-between">
                  <span>Mapped Point p':</span>
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">
                    {response?.transformed_point
                      ? `(${response.transformed_point[0]}, ${response.transformed_point[1]})`
                      : '—'}
                  </span>
                </div>
              </div>
            </div>

            {/* Coordinate Mapping Display */}
            {response && selectedPoint && response.transformed_point && (
              <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 text-xs font-mono space-y-1">
                <div className="flex items-center space-x-1.5 text-slate-700 dark:text-slate-300 font-semibold">
                  <Target className="w-3.5 h-3.5 text-sky-500" />
                  <span>Coordinate Homogeneous Transformation:</span>
                </div>
                <div className="text-slate-600 dark:text-slate-300">
                  <span className="text-sky-700 dark:text-sky-400 font-bold">
                    [{selectedPoint[0]}, {selectedPoint[1]}, 1]ᵀ
                  </span>{' '}
                  <span className="text-slate-400">⟶</span>{' '}
                  <span className="text-emerald-700 dark:text-emerald-400 font-bold">
                    [{response.transformed_point[0]}, {response.transformed_point[1]}, 1]ᵀ
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Combined Mathematical Transformation Matrix */}
          {response && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                <span className="font-semibold text-sm text-slate-900 dark:text-white">
                  Composite Affine Matrix M_composite (3×3 Homogeneous)
                </span>
                <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
                  det(M) = {response.determinant} ({response.is_invertible ? 'Invertible' : 'Singular'})
                </span>
              </div>

              {/* 3x3 Matrix Table */}
              <div className="flex justify-center p-3 bg-slate-50 dark:bg-slate-950 rounded-lg border border-slate-200 dark:border-slate-800 overflow-x-auto">
                <table className="font-mono text-xs text-center border-separate border-spacing-2">
                  <tbody>
                    {response.composite_matrix_3x3.map((row, rIdx) => (
                      <tr key={rIdx}>
                        {row.map((val, cIdx) => (
                          <td
                            key={cIdx}
                            className="px-3 py-1.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 font-bold text-slate-900 dark:text-white"
                          >
                            {val.toFixed(4)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="text-xs text-slate-500 dark:text-slate-400 space-y-2">
                <p>{response.matrix_multiplication_explanation}</p>
              </div>

              {/* Step Matrices Accordion / Cards */}
              {response.steps.length > 0 && (
                <div className="space-y-2 pt-2 border-t border-slate-100 dark:border-slate-800">
                  <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                    Individual Operator Matrices in Composition Chain:
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {response.steps.map((st, i) => (
                      <div
                        key={i}
                        className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-800/40 space-y-1.5 text-xs font-mono"
                      >
                        <div className="flex justify-between items-center">
                          <span className="font-bold text-sky-700 dark:text-sky-300">
                            T_{i + 1}: {st.name}
                          </span>
                          <span className="text-[10px] text-slate-400 uppercase">{st.operation_type}</span>
                        </div>
                        <div className="text-[11px] text-slate-500 dark:text-slate-400 font-sans">
                          {st.description}
                        </div>
                        <div className="p-1.5 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-[11px]">
                          {st.matrix_3x3.map((row, ri) => (
                            <div key={ri} className="flex justify-around">
                              {row.map((val, ci) => (
                                <span key={ci} className="w-16 text-center">
                                  {val.toFixed(2)}
                                </span>
                              ))}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Dedicated Section: Tensor Export System */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 shadow-sm space-y-6">
        <div>
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-sky-500" />
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              Download Raw Image Tensor
            </h3>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Export the complete numerical tensor representation for university-level computing in NumPy,
            SciPy, or PyTorch. Values are exact and never lossy-compressed, normalized, or rounded.
          </p>
        </div>

        {/* Tensor Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* 1. Original RGB */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/40 space-y-3">
            <div className="flex justify-between items-start">
              <span className="font-bold text-xs text-slate-900 dark:text-white">Original RGB Tensor</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-100 dark:bg-sky-950 text-sky-700 dark:text-sky-300">
                uint8
              </span>
            </div>
            <div className="text-xs font-mono text-slate-600 dark:text-slate-300 space-y-1">
              <div>Shape: <span className="font-bold">({metadata.height}, {metadata.width}, 3)</span></div>
              <div>Elements: {metadata.height * metadata.width * 3}</div>
              <div>Size: {formatFileSize(metadata.height * metadata.width * 3)}</div>
            </div>
            <div className="space-y-1.5 pt-2 border-t border-slate-200 dark:border-slate-700">
              <a
                href={getTensorDownloadUrl(metadata.image_id, 'original_rgb', 'npy')}
                className="w-full py-1.5 px-2.5 rounded bg-sky-600 hover:bg-sky-500 text-white text-xs font-mono font-medium flex items-center justify-center space-x-1 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download .npy</span>
              </a>
              <a
                href={getTensorDownloadUrl(metadata.image_id, 'original_rgb', 'csv')}
                className="w-full py-1.5 px-2.5 rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-800 dark:text-slate-200 text-xs font-mono flex items-center justify-center space-x-1 transition-colors"
              >
                <FileSpreadsheet className="w-3.5 h-3.5" />
                <span>Channels (.zip CSV)</span>
              </a>
            </div>
          </div>

          {/* 2. Original Grayscale */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/40 space-y-3">
            <div className="flex justify-between items-start">
              <span className="font-bold text-xs text-slate-900 dark:text-white">Original Grayscale</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
                float64
              </span>
            </div>
            <div className="text-xs font-mono text-slate-600 dark:text-slate-300 space-y-1">
              <div>Shape: <span className="font-bold">({metadata.height}, {metadata.width})</span></div>
              <div>Elements: {metadata.height * metadata.width}</div>
              <div>Size: {formatFileSize(metadata.height * metadata.width * 8)}</div>
            </div>
            <div className="space-y-1.5 pt-2 border-t border-slate-200 dark:border-slate-700">
              <a
                href={getTensorDownloadUrl(metadata.image_id, 'original_grayscale', 'npy')}
                className="w-full py-1.5 px-2.5 rounded bg-sky-600 hover:bg-sky-500 text-white text-xs font-mono font-medium flex items-center justify-center space-x-1 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download .npy</span>
              </a>
              <a
                href={getTensorDownloadUrl(metadata.image_id, 'original_grayscale', 'csv')}
                className="w-full py-1.5 px-2.5 rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-800 dark:text-slate-200 text-xs font-mono flex items-center justify-center space-x-1 transition-colors"
              >
                <FileSpreadsheet className="w-3.5 h-3.5" />
                <span>Matrix (.csv)</span>
              </a>
            </div>
          </div>

          {/* 3. Transformed RGB */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/40 space-y-3">
            <div className="flex justify-between items-start">
              <span className="font-bold text-xs text-slate-900 dark:text-white">Transformed RGB</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300">
                uint8
              </span>
            </div>
            <div className="text-xs font-mono text-slate-600 dark:text-slate-300 space-y-1">
              <div>
                Shape:{' '}
                <span className="font-bold">
                  {tensorMeta?.transformed_rgb
                    ? `(${tensorMeta.transformed_rgb.shape.join(', ')})`
                    : 'Apply transformation'}
                </span>
              </div>
              <div>Elements: {tensorMeta?.transformed_rgb?.total_elements || '—'}</div>
              <div>
                Size:{' '}
                {tensorMeta?.transformed_rgb
                  ? formatFileSize(tensorMeta.transformed_rgb.estimated_size_bytes)
                  : '—'}
              </div>
            </div>
            <div className="space-y-1.5 pt-2 border-t border-slate-200 dark:border-slate-700">
              <a
                href={
                  tensorMeta?.has_transformed
                    ? getTensorDownloadUrl(metadata.image_id, 'transformed_rgb', 'npy')
                    : '#'
                }
                aria-disabled={!tensorMeta?.has_transformed}
                className={`w-full py-1.5 px-2.5 rounded text-xs font-mono font-medium flex items-center justify-center space-x-1 transition-colors ${
                  tensorMeta?.has_transformed
                    ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                    : 'bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed'
                }`}
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download .npy</span>
              </a>
              <a
                href={
                  tensorMeta?.has_transformed
                    ? getTensorDownloadUrl(metadata.image_id, 'transformed_rgb', 'csv')
                    : '#'
                }
                aria-disabled={!tensorMeta?.has_transformed}
                className={`w-full py-1.5 px-2.5 rounded text-xs font-mono flex items-center justify-center space-x-1 transition-colors ${
                  tensorMeta?.has_transformed
                    ? 'bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 text-slate-800 dark:text-slate-200'
                    : 'bg-slate-100 dark:bg-slate-800/50 text-slate-400 cursor-not-allowed'
                }`}
              >
                <FileSpreadsheet className="w-3.5 h-3.5" />
                <span>Channels (.zip CSV)</span>
              </a>
            </div>
          </div>

          {/* 4. All Tensors in NPZ Bundle */}
          <div className="p-4 rounded-xl border border-sky-200 dark:border-sky-900/60 bg-sky-50/50 dark:bg-sky-950/20 space-y-3">
            <div className="flex justify-between items-start">
              <span className="font-bold text-xs text-sky-900 dark:text-sky-300">All Tensors Bundle</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-sky-200 dark:bg-sky-900 text-sky-800 dark:text-sky-200 font-bold">
                .npz
              </span>
            </div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 font-sans">
              Archive containing all available Original & Transformed RGB and Grayscale arrays in NumPy dictionary format.
            </p>
            <div className="pt-3 border-t border-sky-200 dark:border-sky-900/40">
              <a
                href={getTensorDownloadUrl(metadata.image_id, 'all', 'npz')}
                className="w-full py-2 px-3 rounded-lg bg-sky-700 hover:bg-sky-600 text-white text-xs font-mono font-bold flex items-center justify-center space-x-1.5 transition-colors shadow-xs"
              >
                <Download className="w-4 h-4" />
                <span>Download All (.npz)</span>
              </a>
            </div>
          </div>
        </div>

        {/* Verification Example Code Box */}
        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 space-y-2">
          <div className="flex items-center space-x-1.5 text-xs font-mono font-bold text-slate-700 dark:text-slate-300">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
            <span>Python / NumPy Verification Example:</span>
          </div>
          <pre className="font-mono text-xs text-slate-800 dark:text-slate-300 overflow-x-auto whitespace-pre">
{`import numpy as np

# Load downloaded NPY tensor directly into memory
tensor = np.load("${metadata.filename.split('.')[0]}_original_rgb.npy")

print("Tensor Shape:", tensor.shape)  # Expected: (${metadata.height}, ${metadata.width}, 3)
print("Data Type:   ", tensor.dtype)  # Expected: uint8
print("Pixel at (${Math.floor(metadata.width / 2)}, ${Math.floor(metadata.height / 2)}):", tensor[${Math.floor(metadata.height / 2)}, ${Math.floor(metadata.width / 2)}])`}
          </pre>
        </div>
      </div>
    </div>
  );
};

