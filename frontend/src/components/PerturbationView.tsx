import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert,
  Sliders,
  RefreshCw,
  Download,
  Crosshair,
  Info,
  AlertTriangle,
  Eye,
  FileCode,
  FileSpreadsheet,
  Shuffle,
  Activity,
  Flame,
  Database,
  Plus,
  Trash2,
  ArrowUp,
  ArrowDown,
  Upload,
  Layers,
  Check,
  ZoomIn,
} from 'lucide-react';
import type {
  ImageMetadata,
  PerturbationMethod,
  PerturbationStep,
  PerturbationRequest,
  PerturbationResponse,
  CompositionMode,
  TensorMetadataResponse,
} from '../types';
import {
  applyPerturbation,
  getTensorMetadata,
  getTensorDownloadUrl,
} from '../services/api';

interface PerturbationViewProps {
  metadata: ImageMetadata;
  darkMode?: boolean;
}

export const PerturbationView: React.FC<PerturbationViewProps> = ({ metadata }) => {
  // Composition Mode & Target
  const [compositionMode, setCompositionMode] = useState<CompositionMode>('sequential');
  const [baseSource, setBaseSource] = useState<'original' | 'transformed'>('original');
  const [globalBound, setGlobalBound] = useState<number | null>(null);
  const [useGlobalBound, setUseGlobalBound] = useState<boolean>(false);
  const [globalSeed, setGlobalSeed] = useState<number | null>(42);

  // Formal Budget Projection State
  const [budgetType, setBudgetType] = useState<'none' | 'linf' | 'l2' | 'both'>('none');
  const [budgetUnit, setBudgetUnit] = useState<'normalized_0_1' | 'pixel_0_255'>('pixel_0_255');
  const [epsilonLinf, setEpsilonLinf] = useState<number>(10.0);
  const [epsilonL2, setEpsilonL2] = useState<number>(25.0);
  const [l0Threshold, setL0Threshold] = useState<number>(0.00392);

  // Amplification Visualization State
  const [amplificationFactor, setAmplificationFactor] = useState<number>(10);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Operations Pipeline
  const [operations, setOperations] = useState<PerturbationStep[]>([
    {
      id: 'step-1',
      type: 'additive',
      enabled: true,
      weight: 1.0,
      seed: 1234,
      channels: ['R', 'G', 'B'],
      params: { epsilon: 15.0, additive_mode: 'random' },
    },
    {
      id: 'step-2',
      type: 'gaussian',
      enabled: true,
      weight: 1.0,
      seed: 5678,
      channels: ['R', 'G', 'B'],
      params: { sigma: 10.0 },
    },
  ]);

  // Selected tool to add to pipeline
  const [newOpType, setNewOpType] = useState<PerturbationMethod>('multiplicative');

  // Inspector & Coordinates
  const [selectedPoint, setSelectedPoint] = useState<[number, number]>([
    Math.floor(metadata.width / 2),
    Math.floor(metadata.height / 2),
  ]);
  const [submatrixRadius, setSubmatrixRadius] = useState<number>(2); // 5x5 window
  const [submatrixChannel, setSubmatrixChannel] = useState<'R' | 'G' | 'B' | 'GRAY'>('R');

  // Active step preview index (-1 = final result, 0, 1... = intermediate step)
  const [activeStepPreviewIndex, setActiveStepPreviewIndex] = useState<number>(-1);

  // Visualization view mode tab
  const [visualMode, setVisualMode] = useState<
    'side_by_side' | 'heatmap' | 'signed_diff' | 'grayscale' | 'amplified'
  >('side_by_side');

  // Results & States
  const [loading, setLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<PerturbationResponse | null>(null);
  const [tensorMeta, setTensorMeta] = useState<TensorMetadataResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [importSuccessMsg, setImportSuccessMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchTensors = async () => {
    try {
      const meta = await getTensorMetadata(metadata.image_id);
      setTensorMeta(meta);
    } catch {
      // Ignore if not available
    }
  };

  const handleExecute = async () => {
    setLoading(true);
    setErrorMsg(null);

    const req: PerturbationRequest = {
      composition_mode: compositionMode,
      budget: budgetType !== 'none' ? {
        type: budgetType,
        epsilon_linf: (budgetType === 'linf' || budgetType === 'both') ? epsilonLinf : undefined,
        epsilon_l2: (budgetType === 'l2' || budgetType === 'both') ? epsilonL2 : undefined,
        unit: budgetUnit,
      } : undefined,
      budget_unit: budgetUnit,
      l0_threshold: l0Threshold,
      global_bound: useGlobalBound && globalBound !== null ? globalBound : null,
      operations,
      base_source: baseSource,
      seed: globalSeed,
      selected_point: selectedPoint,
      submatrix_radius: submatrixRadius,
    };

    try {
      const res = await applyPerturbation(metadata.image_id, req);
      setResponse(res);
      await fetchTensors();
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to execute perturbation pipeline.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleExecute();
  }, [metadata.image_id, baseSource, compositionMode]);

  // Click-to-inspect coordinate handler on image preview
  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const imgW = e.currentTarget.naturalWidth || metadata.width;
    const imgH = e.currentTarget.naturalHeight || metadata.height;

    const scaleX = imgW / rect.width;
    const scaleY = imgH / rect.height;

    const x = Math.max(0, Math.min(imgW - 1, Math.floor(clickX * scaleX)));
    const y = Math.max(0, Math.min(imgH - 1, Math.floor(clickY * scaleY)));

    setSelectedPoint([x, y]);
  };

  // Re-run inspection when selectedPoint changes
  useEffect(() => {
    if (response) {
      handleExecute();
    }
  }, [selectedPoint[0], selectedPoint[1], submatrixRadius]);

  // Real-time interactive client-side amplified perturbation renderer
  useEffect(() => {
    if (visualMode === 'amplified' && response && canvasRef.current) {
      const origImg = new window.Image();
      const pertImg = new window.Image();
      origImg.crossOrigin = 'anonymous';
      pertImg.crossOrigin = 'anonymous';
      origImg.src = response.original_data_url;
      pertImg.src = response.perturbed_rgb_data_url;

      let loaded = 0;
      const onLoaded = () => {
        loaded++;
        if (loaded === 2 && canvasRef.current) {
          const cvs = canvasRef.current;
          cvs.width = origImg.width;
          cvs.height = origImg.height;
          const ctx = cvs.getContext('2d');
          if (!ctx) return;

          const tCvs1 = document.createElement('canvas');
          tCvs1.width = origImg.width;
          tCvs1.height = origImg.height;
          const tCtx1 = tCvs1.getContext('2d');
          if (!tCtx1) return;
          tCtx1.drawImage(origImg, 0, 0);
          const origData = tCtx1.getImageData(0, 0, origImg.width, origImg.height);

          const tCvs2 = document.createElement('canvas');
          tCvs2.width = pertImg.width;
          tCvs2.height = pertImg.height;
          const tCtx2 = tCvs2.getContext('2d');
          if (!tCtx2) return;
          tCtx2.drawImage(pertImg, 0, 0);
          const pertData = tCtx2.getImageData(0, 0, pertImg.width, pertImg.height);

          const out = ctx.createImageData(origImg.width, origImg.height);
          const k = amplificationFactor;
          for (let i = 0; i < origData.data.length; i += 4) {
            out.data[i] = Math.min(255, Math.abs(pertData.data[i] - origData.data[i]) * k);
            out.data[i + 1] = Math.min(255, Math.abs(pertData.data[i + 1] - origData.data[i + 1]) * k);
            out.data[i + 2] = Math.min(255, Math.abs(pertData.data[i + 2] - origData.data[i + 2]) * k);
            out.data[i + 3] = 255;
          }
          ctx.putImageData(out, 0, 0);
        }
      };
      origImg.onload = onLoaded;
      pertImg.onload = onLoaded;
    }
  }, [visualMode, response, amplificationFactor]);

  // Pipeline operation mutators
  const addOperation = (type: PerturbationMethod) => {
    const defaultParams: Record<string, any> = {};
    if (type === 'additive') {
      defaultParams.epsilon = 15.0;
      defaultParams.additive_mode = 'random';
    } else if (type === 'multiplicative') {
      defaultParams.multiplicative_factor = 0.15;
    } else if (type === 'gaussian') {
      defaultParams.sigma = 15.0;
    } else if (type === 'salt_and_pepper') {
      defaultParams.salt_pepper_prob = 0.05;
      defaultParams.salt_vs_pepper_ratio = 0.5;
    } else if (type === 'inversion') {
      // Negative transform
    } else if (type === 'brightness_contrast') {
      defaultParams.contrast_factor = 1.2;
      defaultParams.brightness_offset = 15.0;
    }

    const newStep: PerturbationStep = {
      id: `step-${Date.now().toString(36)}`,
      type,
      enabled: true,
      weight: 1.0,
      seed: Math.floor(Math.random() * 100000),
      channels: ['R', 'G', 'B'],
      params: defaultParams,
    };

    setOperations([...operations, newStep]);
  };

  const removeOperation = (id: string) => {
    if (operations.length <= 1) {
      setErrorMsg('At least one operation is required in the pipeline.');
      return;
    }
    setOperations(operations.filter((op) => op.id !== id));
    setActiveStepPreviewIndex(-1);
  };

  const toggleOperation = (id: string) => {
    setOperations(
      operations.map((op) => (op.id === id ? { ...op, enabled: !op.enabled } : op))
    );
  };

  const moveOperation = (index: number, direction: 'up' | 'down') => {
    const targetIndex = direction === 'up' ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= operations.length) return;

    const updated = [...operations];
    const [moved] = updated.splice(index, 1);
    updated.splice(targetIndex, 0, moved);
    setOperations(updated);
    setActiveStepPreviewIndex(-1);
  };

  const updateOpParams = (id: string, newParams: Record<string, any>) => {
    setOperations(
      operations.map((op) => (op.id === id ? { ...op, params: { ...op.params, ...newParams } } : op))
    );
  };

  const updateOpWeight = (id: string, weight: number) => {
    const clampedWeight = Math.max(0.0, Math.min(5.0, weight));
    setOperations(
      operations.map((op) => (op.id === id ? { ...op, weight: clampedWeight } : op))
    );
  };

  const updateOpSeed = (id: string, seed: number | null) => {
    setOperations(
      operations.map((op) => (op.id === id ? { ...op, seed } : op))
    );
  };

  const toggleOpChannel = (id: string, ch: 'R' | 'G' | 'B') => {
    setOperations(
      operations.map((op) => {
        if (op.id !== id) return op;
        const exists = op.channels.includes(ch);
        const updatedChannels = exists
          ? op.channels.length > 1
            ? op.channels.filter((c) => c !== ch)
            : op.channels
          : [...op.channels, ch];
        return { ...op, channels: updatedChannels };
      })
    );
  };

  // Pipeline Config Save / Load
  const exportPipelineConfig = () => {
    const configPayload = {
      pipeline_version: '1.1',
      image_filename: metadata.filename,
      composition_mode: compositionMode,
      budget: budgetType !== 'none' ? {
        type: budgetType,
        epsilon_linf: epsilonLinf,
        epsilon_l2: epsilonL2,
        unit: budgetUnit,
      } : null,
      global_bound: useGlobalBound ? globalBound : null,
      seed: globalSeed,
      operations,
      exported_at: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(configPayload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${metadata.filename.split('.')[0]}_perturbation_pipeline.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadProcessedImage = () => {
    if (!response?.perturbed_rgb_data_url) return;
    const baseName = metadata.filename.replace(/\.[^/.]+$/, '');
    const a = document.createElement('a');
    a.href = response.perturbed_rgb_data_url;
    a.download = `photoguard_processed_${baseName}.png`;
    a.click();
  };

  const handleImportConfigFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        if (parsed.composition_mode) {
          setCompositionMode(parsed.composition_mode);
        }
        if (parsed.budget) {
          setBudgetType(parsed.budget.type || 'none');
          if (parsed.budget.unit) setBudgetUnit(parsed.budget.unit);
          if (parsed.budget.epsilon_linf !== undefined) setEpsilonLinf(parsed.budget.epsilon_linf);
          if (parsed.budget.epsilon_l2 !== undefined) setEpsilonL2(parsed.budget.epsilon_l2);
        }
        if (parsed.global_bound !== undefined) {
          setUseGlobalBound(parsed.global_bound !== null);
          setGlobalBound(parsed.global_bound);
        }
        if (parsed.seed !== undefined) {
          setGlobalSeed(parsed.seed);
        }
        if (Array.isArray(parsed.operations) && parsed.operations.length > 0) {
          setOperations(parsed.operations);
        }
        setImportSuccessMsg('Pipeline configuration imported successfully.');
        setTimeout(() => setImportSuccessMsg(null), 4000);
      } catch (err: any) {
        setErrorMsg('Invalid pipeline JSON configuration file.');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const pi = response?.pixel_inspection;
  const metrics = response?.metrics;
  const intermediateSteps = response?.intermediate_steps || [];

  // Determine which image to show in preview
  const isPreviewingIntermediate =
    activeStepPreviewIndex >= 0 && activeStepPreviewIndex < intermediateSteps.length;
  const activeStepData = isPreviewingIntermediate
    ? intermediateSteps[activeStepPreviewIndex]
    : null;
  const activeDisplayUrl = activeStepData
    ? activeStepData.preview_data_url
    : response?.perturbed_rgb_data_url;

  return (
    <div className="space-y-6">
      {/* Title & Phase 4 Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 text-xs font-semibold rounded bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300 border border-amber-300 dark:border-amber-700">
              Phase 4
            </span>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <ShieldAlert className="w-6 h-6 text-amber-600 dark:text-amber-400" />
              Multi-Perturbation Composition Engine
            </h1>
          </div>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Sequential cumulative budget projection, weighted superposition, intermediate tensor tracking & metric analysis.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImportConfigFile}
            accept=".json"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 transition-colors shadow-sm"
          >
            <Upload className="w-3.5 h-3.5 text-slate-500" />
            Import Pipeline (JSON)
          </button>
          <button
            onClick={exportPipelineConfig}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 transition-colors shadow-sm"
          >
            <FileCode className="w-3.5 h-3.5 text-sky-500" />
            Export Pipeline (JSON)
          </button>
          <button
            onClick={handleExecute}
            disabled={loading}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg bg-amber-600 hover:bg-amber-500 text-white shadow-sm transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Recompute Pipeline
          </button>
        </div>
      </div>

      {/* Educational & Research Context Callout */}
      <div className="p-3.5 rounded-lg bg-amber-50/70 dark:bg-amber-950/30 border border-amber-200/80 dark:border-amber-800/50 flex items-start gap-3 text-xs text-amber-800 dark:text-amber-300">
        <Info className="w-4 h-4 shrink-0 mt-0.5 text-amber-600 dark:text-amber-400" />
        <div>
          <span className="font-semibold">Mathematical Research Laboratory: </span>
          This engine evaluates multi-step tensor perturbation models with formal budget projection ($L_\infty$ and $L_2$ balls). Operations are categorized into <span className="font-semibold underline">Bounded Noise</span> (Gaussian, Additive, Multiplicative, Salt & Pepper), <span className="font-semibold underline">Educational Intensity</span> (Inversion), and <span className="font-semibold underline">Photometric</span> (Brightness/Contrast). Note: Standard mathematical perturbations do not guarantee protection against generative AI editing or unauthorized model training.
        </div>
      </div>

      {/* Notifications */}
      {importSuccessMsg && (
        <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/80 flex items-center gap-2 text-xs text-emerald-800 dark:text-emerald-300">
          <Check className="w-4 h-4 text-emerald-600" />
          <span>{importSuccessMsg}</span>
        </div>
      )}

      {errorMsg && (
        <div className="p-3.5 rounded-lg bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/80 flex items-start gap-3 text-xs text-red-800 dark:text-red-300">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-red-600 dark:text-red-400" />
          <div className="flex-1 font-medium">{errorMsg}</div>
        </div>
      )}

      {/* Top Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (2 Cols wide on desktop): Pipeline List & Configuration */}
        <div className="lg:col-span-2 space-y-4">
          {/* Mode Selector Card */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-amber-500" />
                  Composition Pipeline Mode
                </h2>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  Select mathematical combination strategy
                </span>
              </div>

              {/* Mode Radios */}
              <div className="flex p-0.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs">
                <button
                  type="button"
                  onClick={() => setCompositionMode('sequential')}
                  className={`px-3 py-1.5 font-medium rounded-md transition-all ${
                    compositionMode === 'sequential'
                      ? 'bg-white dark:bg-slate-700 text-amber-600 dark:text-amber-300 shadow-sm font-semibold'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                  }`}
                >
                  Sequential (Markov Chain)
                </button>
                <button
                  type="button"
                  onClick={() => setCompositionMode('weighted')}
                  className={`px-3 py-1.5 font-medium rounded-md transition-all ${
                    compositionMode === 'weighted'
                      ? 'bg-white dark:bg-slate-700 text-amber-600 dark:text-amber-300 shadow-sm font-semibold'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                  }`}
                >
                  Weighted Superposition
                </button>
              </div>
            </div>

            {/* Mode Explanation Banner */}
            <div className="p-2.5 rounded bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800 text-xs space-y-1">
              {compositionMode === 'sequential' ? (
                <>
                  <div className="font-semibold text-slate-700 dark:text-slate-300">
                    Mode 1: Sequential Composition —{' '}
                    <span className="font-mono text-amber-600 dark:text-amber-400">
                      I_k = clip(P_k(I_{'{k-1}'}), 0, 255)
                    </span>
                  </div>
                  <p className="text-slate-500 dark:text-slate-400 text-[11px]">
                    Each enabled operation is applied to the output of the preceding stage. Global budget projection is enforced on the cumulative delta relative to base image at every stage.
                  </p>
                </>
              ) : (
                <>
                  <div className="font-semibold text-slate-700 dark:text-slate-300">
                    Mode 2: Weighted Perturbation Combination —{' '}
                    <span className="font-mono text-amber-600 dark:text-amber-400">
                      Δ_comb = ∑ w_i Δ_i, I_final = clip(I + Δ_comb, 0, 255)
                    </span>
                  </div>
                  <p className="text-slate-500 dark:text-slate-400 text-[11px]">
                    Each enabled operation calculates its delta field relative to the <span className="font-semibold">same base image</span>. Perturbations are scaled by weights (0–5), linearly combined, and projected onto the configured budget ball.
                  </p>
                </>
              )}
            </div>

            {/* Target Base Source & Global Seed */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-slate-100 dark:border-slate-800">
              <div>
                <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Target Base Source
                </label>
                <div className="grid grid-cols-2 gap-1.5">
                  <button
                    type="button"
                    onClick={() => setBaseSource('original')}
                    className={`px-2.5 py-1.5 text-xs font-medium rounded border transition-all ${
                      baseSource === 'original'
                        ? 'border-amber-500 bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300'
                        : 'border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 text-slate-700 dark:text-slate-300'
                    }`}
                  >
                    Original Image
                  </button>
                  <button
                    type="button"
                    onClick={() => setBaseSource('transformed')}
                    disabled={!tensorMeta?.has_transformed}
                    className={`px-2.5 py-1.5 text-xs font-medium rounded border transition-all ${
                      baseSource === 'transformed'
                        ? 'border-amber-500 bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300'
                        : 'border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 text-slate-700 dark:text-slate-300'
                    } ${!tensorMeta?.has_transformed ? 'opacity-40 cursor-not-allowed' : ''}`}
                  >
                    Phase 3 Transformed
                  </button>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-medium text-slate-500 dark:text-slate-400">
                    Pipeline Random Seed
                  </label>
                  <button
                    type="button"
                    onClick={() => setGlobalSeed(Math.floor(Math.random() * 10000))}
                    className="text-[11px] text-amber-600 dark:text-amber-400 hover:underline inline-flex items-center gap-1"
                  >
                    <Shuffle className="w-3 h-3" />
                    Randomize
                  </button>
                </div>
                <input
                  type="number"
                  value={globalSeed ?? ''}
                  onChange={(e) => setGlobalSeed(e.target.value ? parseInt(e.target.value) : null)}
                  placeholder="None (non-deterministic)"
                  className="w-full px-3 py-1.5 text-xs font-mono rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200"
                />
              </div>
            </div>
          </div>

          {/* Formal Perturbation Budget Card */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4 text-sky-500" />
                  Perturbation Budget Projection (L_∞ & L_2 Balls)
                </h2>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  Mathematical constraint bounding cumulative tensor deviation relative to base image
                </span>
              </div>
              <div className="flex p-0.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs">
                {(['none', 'linf', 'l2', 'both'] as const).map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setBudgetType(t)}
                    className={`px-2.5 py-1 font-medium rounded-md transition-all ${
                      budgetType === t
                        ? 'bg-white dark:bg-slate-700 text-sky-600 dark:text-sky-300 shadow-sm font-semibold'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                    }`}
                  >
                    {t === 'none' ? 'None' : t === 'linf' ? 'L_∞ Ball' : t === 'l2' ? 'L_2 Ball' : 'Both (L_∞ + L_2)'}
                  </button>
                ))}
              </div>
            </div>

            {budgetType !== 'none' && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 border-t border-slate-100 dark:border-slate-800 text-xs">
                <div>
                  <label className="block text-slate-500 dark:text-slate-400 font-medium mb-1">Parameter Unit</label>
                  <div className="grid grid-cols-2 gap-1">
                    <button
                      type="button"
                      onClick={() => {
                        setBudgetUnit('pixel_0_255');
                        setEpsilonLinf(10.0);
                        setEpsilonL2(25.0);
                      }}
                      className={`px-2 py-1 rounded text-xs border transition-all ${
                        budgetUnit === 'pixel_0_255'
                          ? 'border-sky-500 bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 font-semibold'
                          : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      Pixel [0, 255]
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setBudgetUnit('normalized_0_1');
                        setEpsilonLinf(8.0 / 255.0);
                        setEpsilonL2(1.0);
                      }}
                      className={`px-2 py-1 rounded text-xs border transition-all ${
                        budgetUnit === 'normalized_0_1'
                          ? 'border-sky-500 bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 font-semibold'
                          : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      Float [0, 1]
                    </button>
                  </div>
                </div>

                {(budgetType === 'linf' || budgetType === 'both') && (
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-slate-500 dark:text-slate-400 font-medium">L_∞ Bound ε_∞</span>
                      <span className="font-mono text-sky-600 dark:text-sky-400 font-bold">
                        {budgetUnit === 'normalized_0_1' ? epsilonLinf.toFixed(4) : epsilonLinf.toFixed(1)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={budgetUnit === 'normalized_0_1' ? '0.001' : '0.5'}
                      max={budgetUnit === 'normalized_0_1' ? '0.2' : '50.0'}
                      step={budgetUnit === 'normalized_0_1' ? '0.001' : '0.5'}
                      value={epsilonLinf}
                      onChange={(e) => setEpsilonLinf(parseFloat(e.target.value))}
                      className="w-full accent-sky-600"
                    />
                  </div>
                )}

                {(budgetType === 'l2' || budgetType === 'both') && (
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-slate-500 dark:text-slate-400 font-medium">L_2 Bound ε_2</span>
                      <span className="font-mono text-sky-600 dark:text-sky-400 font-bold">
                        {budgetUnit === 'normalized_0_1' ? epsilonL2.toFixed(3) : epsilonL2.toFixed(1)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={budgetUnit === 'normalized_0_1' ? '0.05' : '1.0'}
                      max={budgetUnit === 'normalized_0_1' ? '5.0' : '150.0'}
                      step={budgetUnit === 'normalized_0_1' ? '0.05' : '1.0'}
                      value={epsilonL2}
                      onChange={(e) => setEpsilonL2(parseFloat(e.target.value))}
                      className="w-full accent-sky-600"
                    />
                  </div>
                )}

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-500 dark:text-slate-400 font-medium">L_0 Threshold τ</span>
                    <span className="font-mono text-sky-600 dark:text-sky-400 font-bold">
                      {l0Threshold.toFixed(4)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.0001"
                    max="0.05"
                    step="0.0005"
                    value={l0Threshold}
                    onChange={(e) => setL0Threshold(parseFloat(e.target.value))}
                    className="w-full accent-sky-600"
                  />
                </div>
              </div>
            )}

            {response?.budget_report && (
              <div className="p-2.5 rounded bg-sky-50/70 dark:bg-sky-950/30 border border-sky-200/60 dark:border-sky-800/40 text-xs flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sky-900 dark:text-sky-300">Projection Status:</span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      response.budget_report.projected
                        ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300'
                        : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300'
                    }`}
                  >
                    {response.budget_report.projected ? 'Projection Applied (Clamped)' : 'Within Budget (No Scaling)'}
                  </span>
                </div>
                <div className="font-mono text-[11px] text-slate-600 dark:text-slate-400">
                  Pre: L_∞={response.budget_report.pre_projection_linf?.toFixed(1)}, L_2={response.budget_report.pre_projection_l2?.toFixed(1)} →
                  Post: L_∞={response.budget_report.post_projection_linf?.toFixed(1)}, L_2={response.budget_report.post_projection_l2?.toFixed(1)}
                </div>
              </div>
            )}
          </div>

          {/* Pipeline Operations List */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-amber-500" />
                  Configured Operations ({operations.length})
                </h2>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {compositionMode === 'sequential'
                    ? 'Applied sequentially with cumulative budget constraint'
                    : 'Combined simultaneously with respective weights (0–5)'}
                </span>
              </div>

              {/* Add Operation Dropdown */}
              <div className="flex items-center gap-2">
                <select
                  value={newOpType}
                  onChange={(e) => setNewOpType(e.target.value as PerturbationMethod)}
                  className="px-2.5 py-1.5 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                >
                  <option value="additive">Additive Noise (||Δ||_∞ ≤ ε)</option>
                  <option value="gaussian">Gaussian White Noise (σ)</option>
                  <option value="multiplicative">Multiplicative Scaling (α)</option>
                  <option value="salt_and_pepper">Salt & Pepper Impulse (p, r)</option>
                  <option value="brightness_contrast">Brightness & Contrast (c, b)</option>
                  <option value="inversion">Pixel Inversion (Negative)</option>
                </select>
                <button
                  type="button"
                  onClick={() => addOperation(newOpType)}
                  className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-semibold rounded-lg bg-amber-600 hover:bg-amber-500 text-white shadow-sm transition-all"
                >
                  <Plus className="w-3.5 h-3.5" />
                  Add Step
                </button>
              </div>
            </div>

            {/* List of Step Cards */}
            <div className="space-y-3">
              {operations.map((op, index) => {
                const isEducational =
                  op.type === 'inversion' || op.type === 'brightness_contrast';

                return (
                  <div
                    key={op.id}
                    className={`p-3.5 rounded-lg border transition-all ${
                      op.enabled
                        ? 'border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-800/40'
                        : 'border-slate-200/50 dark:border-slate-800/40 bg-slate-100/40 dark:bg-slate-900/40 opacity-60'
                    }`}
                  >
                    {/* Header: Step Index, Type, Badges, Up/Down, Remove */}
                    <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-200/60 dark:border-slate-800/60">
                      <div className="flex items-center gap-2">
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={op.enabled}
                            onChange={() => toggleOperation(op.id)}
                            className="rounded text-amber-600 focus:ring-amber-500"
                          />
                          <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                            {index + 1}.{' '}
                            {op.type === 'additive'
                              ? 'Bounded Additive Noise'
                              : op.type === 'gaussian'
                              ? 'Gaussian Noise'
                              : op.type === 'multiplicative'
                              ? 'Multiplicative Scaling'
                              : op.type === 'salt_and_pepper'
                              ? 'Salt & Pepper Impulse'
                              : op.type === 'brightness_contrast'
                              ? 'Brightness & Contrast'
                              : 'Pixel Inversion (Negative)'}
                          </span>
                        </label>

                        {/* Category Badge */}
                        <span
                          className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${
                            isEducational
                              ? 'bg-purple-100 dark:bg-purple-950/60 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800'
                              : 'bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800'
                          }`}
                        >
                          {isEducational ? 'Educational / Photometric' : 'Bounded Noise'}
                        </span>
                      </div>

                      {/* Controls: Reorder, Step Seed & Remove */}
                      <div className="flex items-center gap-2">
                        {/* Step Seed */}
                        <div className="flex items-center gap-1 text-[11px]">
                          <span className="text-slate-400">Seed:</span>
                          <input
                            type="number"
                            value={op.seed ?? ''}
                            placeholder="Inherit"
                            onChange={(e) =>
                              updateOpSeed(op.id, e.target.value ? parseInt(e.target.value) : null)
                            }
                            className="w-16 px-1.5 py-0.5 text-[11px] font-mono rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                          />
                          <button
                            type="button"
                            onClick={() => updateOpSeed(op.id, Math.floor(Math.random() * 100000))}
                            title="Randomize Step Seed"
                            className="p-1 rounded text-slate-400 hover:text-amber-600"
                          >
                            <Shuffle className="w-3 h-3" />
                          </button>
                        </div>

                        <div className="flex items-center gap-0.5">
                          <button
                            type="button"
                            onClick={() => moveOperation(index, 'up')}
                            disabled={index === 0}
                            title="Move step up"
                            className="p-1 rounded text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-30"
                          >
                            <ArrowUp className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => moveOperation(index, 'down')}
                            disabled={index === operations.length - 1}
                            title="Move step down"
                            className="p-1 rounded text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-30"
                          >
                            <ArrowDown className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => removeOperation(op.id)}
                            title="Remove step"
                            className="p-1 rounded text-red-500 hover:bg-red-50 dark:hover:bg-red-950/40 ml-1"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Step Body: Parameters & Weight */}
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                      {/* Weight Control (Range 0 to 5) */}
                      {compositionMode === 'weighted' && (
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-slate-500 dark:text-slate-400 font-medium">
                              Weight w_i (0–5)
                            </span>
                            <span className="font-mono text-amber-600 dark:text-amber-400 font-bold">
                              {op.weight.toFixed(2)}
                            </span>
                          </div>
                          <input
                            type="range"
                            min="0.0"
                            max="5.0"
                            step="0.05"
                            value={op.weight}
                            onChange={(e) => updateOpWeight(op.id, parseFloat(e.target.value))}
                            className="w-full accent-amber-600"
                          />
                        </div>
                      )}

                      {/* Specific Parameter Sliders */}
                      {op.type === 'additive' && (
                        <>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-slate-500 dark:text-slate-400 font-medium">Bound ε</span>
                              <span className="font-mono font-bold text-amber-600 dark:text-amber-400">
                                {Number(op.params.epsilon ?? 15).toFixed(1)}
                              </span>
                            </div>
                            <input
                              type="range"
                              min="1"
                              max="100"
                              step="1"
                              value={op.params.epsilon ?? 15}
                              onChange={(e) => updateOpParams(op.id, { epsilon: parseFloat(e.target.value) })}
                              className="w-full accent-amber-600"
                            />
                          </div>
                          <div>
                            <span className="text-slate-500 dark:text-slate-400 font-medium block mb-1">
                              Mode
                            </span>
                            <select
                              value={op.params.additive_mode ?? 'random'}
                              onChange={(e) => updateOpParams(op.id, { additive_mode: e.target.value })}
                              className="w-full px-2 py-1 text-xs rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200"
                            >
                              <option value="random">Random [-ε, +ε]</option>
                              <option value="uniform">Uniform (+ε)</option>
                              <option value="positive_const">Constant (+ε)</option>
                              <option value="negative_const">Constant (-ε)</option>
                            </select>
                          </div>
                        </>
                      )}

                      {op.type === 'gaussian' && (
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-slate-500 dark:text-slate-400 font-medium">Sigma σ</span>
                            <span className="font-mono font-bold text-amber-600 dark:text-amber-400">
                              {Number(op.params.sigma ?? 15).toFixed(1)}
                            </span>
                          </div>
                          <input
                            type="range"
                            min="1"
                            max="80"
                            step="1"
                            value={op.params.sigma ?? 15}
                            onChange={(e) => updateOpParams(op.id, { sigma: parseFloat(e.target.value) })}
                            className="w-full accent-amber-600"
                          />
                        </div>
                      )}

                      {op.type === 'multiplicative' && (
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-slate-500 dark:text-slate-400 font-medium">Factor α</span>
                            <span className="font-mono font-bold text-amber-600 dark:text-amber-400">
                              ±{(Number(op.params.multiplicative_factor ?? 0.15) * 100).toFixed(0)}%
                            </span>
                          </div>
                          <input
                            type="range"
                            min="0.01"
                            max="0.80"
                            step="0.01"
                            value={op.params.multiplicative_factor ?? 0.15}
                            onChange={(e) =>
                              updateOpParams(op.id, { multiplicative_factor: parseFloat(e.target.value) })
                            }
                            className="w-full accent-amber-600"
                          />
                        </div>
                      )}

                      {op.type === 'salt_and_pepper' && (
                        <>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-slate-500 dark:text-slate-400 font-medium">Probability p</span>
                              <span className="font-mono font-bold text-amber-600 dark:text-amber-400">
                                {Number(op.params.salt_pepper_prob ?? 0.05).toFixed(3)}
                              </span>
                            </div>
                            <input
                              type="range"
                              min="0.005"
                              max="0.30"
                              step="0.005"
                              value={op.params.salt_pepper_prob ?? 0.05}
                              onChange={(e) =>
                                updateOpParams(op.id, { salt_pepper_prob: parseFloat(e.target.value) })
                              }
                              className="w-full accent-amber-600"
                            />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-slate-500 dark:text-slate-400 font-medium">Salt Ratio r</span>
                              <span className="font-mono font-bold text-amber-600 dark:text-amber-400">
                                {(Number(op.params.salt_vs_pepper_ratio ?? 0.5) * 100).toFixed(0)}% Salt
                              </span>
                            </div>
                            <input
                              type="range"
                              min="0.0"
                              max="1.0"
                              step="0.05"
                              value={op.params.salt_vs_pepper_ratio ?? 0.5}
                              onChange={(e) =>
                                updateOpParams(op.id, { salt_vs_pepper_ratio: parseFloat(e.target.value) })
                              }
                              className="w-full accent-amber-600"
                            />
                          </div>
                        </>
                      )}

                      {op.type === 'brightness_contrast' && (
                        <>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-slate-500 dark:text-slate-400 font-medium">Contrast c</span>
                              <span className="font-mono font-bold text-amber-600 dark:text-amber-400">
                                {Number(op.params.contrast_factor ?? 1.2).toFixed(2)}x
                              </span>
                            </div>
                            <input
                              type="range"
                              min="0.2"
                              max="2.5"
                              step="0.05"
                              value={op.params.contrast_factor ?? 1.2}
                              onChange={(e) =>
                                updateOpParams(op.id, { contrast_factor: parseFloat(e.target.value) })
                              }
                              className="w-full accent-amber-600"
                            />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-slate-500 dark:text-slate-400 font-medium">Brightness b</span>
                              <span className="font-mono font-bold text-amber-600 dark:text-amber-400">
                                {Number(op.params.brightness_offset ?? 15) > 0 ? '+' : ''}
                                {Number(op.params.brightness_offset ?? 15).toFixed(0)}
                              </span>
                            </div>
                            <input
                              type="range"
                              min="-80"
                              max="80"
                              step="1"
                              value={op.params.brightness_offset ?? 15}
                              onChange={(e) =>
                                updateOpParams(op.id, { brightness_offset: parseFloat(e.target.value) })
                              }
                              className="w-full accent-amber-600"
                            />
                          </div>
                        </>
                      )}

                      {op.type === 'inversion' && (
                        <div className="text-[11px] text-slate-500 italic flex items-center">
                          Direct photographic negative: I' = 255 - I
                        </div>
                      )}

                      {/* Channels Checkboxes */}
                      <div>
                        <span className="text-slate-500 dark:text-slate-400 font-medium block mb-1">
                          Channels
                        </span>
                        <div className="flex gap-1">
                          {(['R', 'G', 'B'] as const).map((ch) => {
                            const isChActive = op.channels.includes(ch);
                            return (
                              <button
                                key={ch}
                                type="button"
                                onClick={() => toggleOpChannel(op.id, ch)}
                                className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-all ${
                                  isChActive
                                    ? ch === 'R'
                                      ? 'border-red-400 bg-red-100 text-red-700'
                                      : ch === 'G'
                                      ? 'border-emerald-400 bg-emerald-100 text-emerald-700'
                                      : 'border-blue-400 bg-blue-100 text-blue-700'
                                    : 'border-slate-200 dark:border-slate-700 text-slate-400'
                                }`}
                              >
                                {ch}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column (1 Col): Numerical Tensor Download Center & Formula */}
        <div className="space-y-4">
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Database className="w-4 h-4 text-sky-500" />
              Lossless Numerical Tensor Export
            </h2>

            <div className="p-2.5 rounded bg-amber-50/60 dark:bg-amber-950/30 border border-amber-200/60 dark:border-amber-800/40 text-[11px] text-amber-800 dark:text-amber-300">
              <span className="font-semibold">Binary Tensor Notice: </span>
              NumPy <code className="font-mono font-bold">.npy</code> and{' '}
              <code className="font-mono font-bold">.npz</code> are binary files. Open using Python:
              <pre className="mt-1 font-mono text-[10px] p-1 rounded bg-amber-100/60 dark:bg-amber-900/40">
                import numpy as np; data = np.load('tensors.npz')
              </pre>
            </div>

            <div className="space-y-2">
              {/* ── Download Processed Image (PNG) ── */}
              <div className="pb-2 border-b border-slate-100 dark:border-slate-800">
                <button
                  onClick={handleDownloadProcessedImage}
                  disabled={!response?.perturbed_rgb_data_url}
                  className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-300 dark:disabled:bg-slate-700 disabled:cursor-not-allowed text-white shadow-sm transition-colors"
                  title={
                    response?.perturbed_rgb_data_url
                      ? 'Download the perturbed image as a lossless PNG'
                      : 'Apply perturbations first to enable download'
                  }
                >
                  <Download className="w-3.5 h-3.5" />
                  Download Processed Image (.png)
                </button>
                {!response?.perturbed_rgb_data_url && (
                  <p className="mt-1 text-[10px] text-slate-400 dark:text-slate-500 text-center">
                    Apply perturbations to enable PNG download
                  </p>
                )}
              </div>

              <a
                href={getTensorDownloadUrl(metadata.image_id, 'all', 'npz')}
                download
                className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-lg bg-sky-600 hover:bg-sky-500 text-white shadow-sm transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                Download Complete NPZ Bundle (.npz)
              </a>

              <a
                href={getTensorDownloadUrl(metadata.image_id, 'metadata', 'json')}
                download
                className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 text-slate-700 dark:text-slate-200 transition-colors"
              >
                <FileCode className="w-3.5 h-3.5 text-amber-500" />
                Experiment Metadata (.json)
              </a>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <a
                  href={getTensorDownloadUrl(metadata.image_id, 'perturbed_rgb', 'npy')}
                  download
                  className="inline-flex items-center justify-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 text-slate-700 dark:text-slate-200"
                >
                  <Download className="w-3 h-3 text-sky-500" />
                  Final RGB (.npy)
                </a>

                <a
                  href={getTensorDownloadUrl(metadata.image_id, 'difference', 'npy')}
                  download
                  className="inline-flex items-center justify-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 text-slate-700 dark:text-slate-200"
                >
                  <Download className="w-3 h-3 text-amber-500" />
                  Delta Δ (.npy)
                </a>

                <a
                  href={getTensorDownloadUrl(metadata.image_id, 'difference', 'csv')}
                  download
                  className="inline-flex items-center justify-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 text-slate-700 dark:text-slate-200"
                >
                  <FileSpreadsheet className="w-3 h-3 text-emerald-500" />
                  Delta (.csv)
                </a>

                <a
                  href={getTensorDownloadUrl(metadata.image_id, 'all', 'csv')}
                  download
                  className="inline-flex items-center justify-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 text-slate-700 dark:text-slate-200"
                >
                  <Download className="w-3 h-3 text-purple-500" />
                  All CSVs (.zip)
                </a>
              </div>
            </div>

            {/* Active Math Model */}
            {response && (
              <div className="pt-3 border-t border-slate-100 dark:border-slate-800">
                <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                  Active Mathematical Pipeline:
                </div>
                <div className="p-2.5 rounded bg-slate-100 dark:bg-slate-800 font-mono text-xs text-amber-700 dark:text-amber-300 overflow-x-auto">
                  {response.mathematical_formula}
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1.5">
                  {response.method_explanation}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Step-by-Step Intermediate Stage Scrubber */}
      {intermediateSteps.length > 0 && (
        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Eye className="w-4 h-4 text-amber-500" />
              Pipeline Intermediate Stage Scrubber
            </h2>
            <span className="text-xs text-slate-400 font-mono">
              Click any stage to preview intermediate tensor output
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setActiveStepPreviewIndex(-2)} // -2 = base original
              className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${
                activeStepPreviewIndex === -2
                  ? 'border-sky-500 bg-sky-50 dark:bg-sky-950/60 text-sky-700 dark:text-sky-300 font-bold shadow-sm'
                  : 'border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-50'
              }`}
            >
              Base Input I_0
            </button>

            {intermediateSteps.map((s, idx) => (
              <button
                key={s.step_id}
                type="button"
                onClick={() => setActiveStepPreviewIndex(idx)}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${
                  activeStepPreviewIndex === idx
                    ? 'border-amber-500 bg-amber-50 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 font-bold shadow-sm'
                    : 'border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-50'
                }`}
              >
                Step {s.step_index}: {s.name}
              </button>
            ))}

            <button
              type="button"
              onClick={() => setActiveStepPreviewIndex(-1)} // -1 = final result
              className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${
                activeStepPreviewIndex === -1
                  ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 font-bold shadow-sm'
                  : 'border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-50'
              }`}
            >
              Final Result I_final
            </button>
          </div>
        </div>
      )}

      {/* Visual Previews, Difference Heatmap & Amplified Perturbation */}
      <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-amber-500" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Visual Inspection & Difference Analysis
            </h2>
          </div>

          {/* Mode Switcher */}
          <div className="flex p-0.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs">
            {[
              { id: 'side_by_side', label: 'Side-by-Side' },
              { id: 'heatmap', label: 'Error Heatmap |Δ|' },
              { id: 'signed_diff', label: 'Signed Diff Δ' },
              { id: 'amplified', label: 'Amplified Perturbation (|Δ| × k)' },
              { id: 'grayscale', label: 'Perturbed Grayscale' },
            ].map((v) => (
              <button
                key={v.id}
                type="button"
                onClick={() => setVisualMode(v.id as any)}
                className={`px-3 py-1.5 font-medium rounded-md transition-all ${
                  visualMode === v.id
                    ? 'bg-white dark:bg-slate-700 text-amber-600 dark:text-amber-300 shadow-sm font-semibold'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>

        {/* Display Content */}
        {response && (
          <div>
            {visualMode === 'side_by_side' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Base Source */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-700 dark:text-slate-300">
                      Base Input I_0 ({response.base_source === 'transformed' ? 'Transformed' : 'Original'})
                    </span>
                    <span className="text-slate-400 font-mono">
                      {response.width} × {response.height} × {response.channels}
                    </span>
                  </div>
                  <div className="relative border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[300px]">
                    <img
                      src={response.original_data_url}
                      alt="Base Original"
                      className="max-h-[420px] max-w-full object-contain"
                    />
                  </div>
                </div>

                {/* Displayed Image */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-amber-700 dark:text-amber-300 flex items-center gap-1.5">
                      <Crosshair className="w-3.5 h-3.5 text-amber-500" />
                      {isPreviewingIntermediate
                        ? `Intermediate Stage ${activeStepData?.step_index}: ${activeStepData?.name}`
                        : activeStepPreviewIndex === -2
                        ? 'Base Input I_0'
                        : 'Final Perturbed Image I_final'}
                    </span>
                    <span className="text-slate-400 font-mono">
                      Inspect: ({selectedPoint[0]}, {selectedPoint[1]})
                    </span>
                  </div>
                  <div className="relative border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[300px] cursor-crosshair">
                    <img
                      src={
                        activeStepPreviewIndex === -2
                          ? response.original_data_url
                          : activeDisplayUrl || response.perturbed_rgb_data_url
                      }
                      alt="Preview"
                      onClick={handleImageClick}
                      className="max-h-[420px] max-w-full object-contain"
                    />
                  </div>
                </div>
              </div>
            )}

            {visualMode === 'heatmap' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                    <Flame className="w-3.5 h-3.5 text-red-500" />
                    Absolute Error Heatmap |Δ| = |I_final - I_base| (INFERNO Colormap)
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-slate-400">Zero Error</span>
                    <div className="w-24 h-2.5 rounded bg-gradient-to-r from-black via-red-600 via-amber-400 to-white" />
                    <span className="text-[11px] text-slate-400 font-bold">Max Error</span>
                  </div>
                </div>
                <div className="border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[340px]">
                  <img
                    src={response.difference_heatmap_data_url}
                    alt="Error Heatmap"
                    className="max-h-[460px] max-w-full object-contain"
                  />
                </div>
              </div>
            )}

            {visualMode === 'signed_diff' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-700 dark:text-slate-300">
                    Signed Perturbation Δ = I_final - I_base (Neutral Gray = 128)
                  </span>
                  <span className="text-[11px] text-slate-400 font-mono">
                    Dark: Negative Shift | Gray: Zero | Bright: Positive Shift
                  </span>
                </div>
                <div className="border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[340px]">
                  <img
                    src={response.signed_difference_data_url}
                    alt="Signed Difference"
                    className="max-h-[460px] max-w-full object-contain"
                  />
                </div>
              </div>
            )}

            {visualMode === 'amplified' && (
              <div className="space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                  <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                    <ZoomIn className="w-4 h-4 text-amber-500" />
                    Amplified Absolute Perturbation Tensor: |Δ| × {amplificationFactor}
                  </span>
                  <div className="flex items-center gap-3">
                    <span className="text-slate-500 dark:text-slate-400 font-medium">Amplification Multiplier:</span>
                    <input
                      type="range"
                      min="1"
                      max="50"
                      step="1"
                      value={amplificationFactor}
                      onChange={(e) => setAmplificationFactor(parseInt(e.target.value) || 1)}
                      className="w-32 accent-amber-600"
                    />
                    <span className="font-mono font-bold text-amber-600 dark:text-amber-400 w-8">
                      {amplificationFactor}×
                    </span>
                  </div>
                </div>
                <div className="border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[340px] p-2">
                  <canvas ref={canvasRef} className="max-h-[460px] max-w-full object-contain" />
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400 text-center">
                  Subtle, imperceptible perturbations are scaled linearly by {amplificationFactor}× against a black baseline, exposing structural noise spatial distribution.
                </div>
              </div>
            )}

            {visualMode === 'grayscale' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-700 dark:text-slate-300">
                    Final Perturbed Grayscale Matrix Y' = 0.299R' + 0.587G' + 0.114B'
                  </span>
                </div>
                <div className="border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[340px]">
                  <img
                    src={response.perturbed_grayscale_data_url}
                    alt="Perturbed Grayscale"
                    className="max-h-[460px] max-w-full object-contain"
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Comprehensive Mathematical Quality Metrics */}
      {metrics && (
        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-500" />
              Final Perturbation Metrics & Norms
            </h2>
            <div className="flex items-center gap-3">
              {metrics.lpips !== null && metrics.lpips !== undefined && (
                <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-indigo-100 text-indigo-800 dark:bg-indigo-900/40 dark:text-indigo-300">
                  LPIPS: {metrics.lpips.toFixed(4)}
                </span>
              )}
              <span className="text-xs text-slate-400 font-mono">
                Evaluated across {metrics.total_pixels_count.toLocaleString()} tensor elements
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
            {/* MSE */}
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">Mean Squared Error</div>
              <div className="text-lg font-bold text-slate-800 dark:text-slate-100 font-mono">
                {metrics.mse.toFixed(2)}
              </div>
              <div className="text-[10px] text-slate-400 font-mono">MSE = 1/N ∑(I'-I)²</div>
            </div>

            {/* PSNR */}
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">PSNR (dB)</div>
              <div className="text-lg font-bold text-slate-800 dark:text-slate-100 font-mono">
                {metrics.psnr > 100 ? '∞' : metrics.psnr.toFixed(2)}
              </div>
              <div className="text-[10px] text-slate-400 font-mono">10 log₁₀(255²/MSE)</div>
            </div>

            {/* SSIM */}
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">SSIM Index</div>
              <div
                className={`text-lg font-bold font-mono ${
                  metrics.ssim >= 0.9
                    ? 'text-emerald-600 dark:text-emerald-400'
                    : metrics.ssim >= 0.7
                    ? 'text-amber-600 dark:text-amber-400'
                    : 'text-red-600 dark:text-red-400'
                }`}
              >
                {metrics.ssim.toFixed(4)}
              </div>
              <div className="text-[10px] text-slate-400 font-mono">Structural [0, 1]</div>
            </div>

            {/* MAE */}
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">MAE</div>
              <div className="text-lg font-bold text-slate-800 dark:text-slate-100 font-mono">
                {metrics.mae.toFixed(2)}
              </div>
              <div className="text-[10px] text-slate-400 font-mono">1/N ∑ |I'-I|</div>
            </div>

            {/* L_inf Norm */}
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">||Δ||_∞ Norm</div>
              <div className="text-lg font-bold text-amber-600 dark:text-amber-400 font-mono">
                {metrics.linf_norm.toFixed(1)}
              </div>
              <div className="text-[10px] text-slate-400 font-mono">Max absolute error</div>
            </div>

            {/* L_2 Norm */}
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">||Δ||₂ (Frobenius)</div>
              <div className="text-lg font-bold text-slate-800 dark:text-slate-100 font-mono">
                {metrics.l2_norm >= 1000 ? metrics.l2_norm.toExponential(2) : metrics.l2_norm.toFixed(1)}
              </div>
              <div className="text-[10px] text-slate-400 font-mono">Euclidean distance</div>
            </div>

            {/* Standard Deviation */}
            <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">Std Deviation σ_Δ</div>
              <div className="text-lg font-bold text-slate-800 dark:text-slate-100 font-mono">
                {metrics.std_diff !== undefined ? metrics.std_diff.toFixed(2) : '0.00'}
              </div>
              <div className="text-[10px] text-slate-400 font-mono">Noise dispersion</div>
            </div>
          </div>

          {/* Dual Card: Per-Channel Breakdown & Quantization Distortion / L0 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-2">
            {/* Per-Channel Metrics Table */}
            {metrics.per_channel && (
              <div className="border border-slate-200 dark:border-slate-800 rounded-lg p-3 bg-slate-50/50 dark:bg-slate-800/40 space-y-2">
                <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
                  <span>Per-Channel Degradation Breakdown</span>
                  <span className="text-[10px] font-mono text-slate-400">R, G, B Channels</span>
                </div>
                <table className="w-full text-[11px]">
                  <thead className="text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-700">
                    <tr>
                      <th className="py-1 text-left">Ch</th>
                      <th className="py-1 text-right">||Δ||_∞</th>
                      <th className="py-1 text-right">||Δ||₂</th>
                      <th className="py-1 text-right">MAE</th>
                      <th className="py-1 text-right">MSE</th>
                      <th className="py-1 text-right">Mean</th>
                      <th className="py-1 text-right">Std</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono">
                    {(['R', 'G', 'B'] as const).map((ch) => {
                      const cData = metrics.per_channel?.[ch];
                      if (!cData) return null;
                      return (
                        <tr key={ch}>
                          <td className="py-1 font-bold text-amber-600 dark:text-amber-400">{ch}</td>
                          <td className="py-1 text-right">{cData.linf_norm.toFixed(1)}</td>
                          <td className="py-1 text-right">{cData.l2_norm.toFixed(1)}</td>
                          <td className="py-1 text-right">{cData.mae.toFixed(2)}</td>
                          <td className="py-1 text-right">{cData.mse.toFixed(1)}</td>
                          <td className="py-1 text-right">{cData.mean_diff > 0 ? `+${cData.mean_diff.toFixed(2)}` : cData.mean_diff.toFixed(2)}</td>
                          <td className="py-1 text-right">{cData.std_diff.toFixed(2)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Quantization Distortion & L0 Dual-Measurement */}
            <div className="border border-slate-200 dark:border-slate-800 rounded-lg p-3 bg-slate-50/50 dark:bg-slate-800/40 space-y-3">
              <div className="flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-slate-300">
                <span>Quantization Distortion & L_0 Norm</span>
                <span className="text-[10px] font-mono text-slate-400">E_quant = Δ_uint8 - Δ_cont</span>
              </div>

              {metrics.quantization_distortion && (
                <div className="grid grid-cols-4 gap-2 text-center text-xs font-mono">
                  <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
                    <div className="text-[10px] text-slate-400 font-sans">Quant MAE</div>
                    <div className="font-bold text-slate-800 dark:text-slate-200">
                      {metrics.quantization_distortion.mae.toFixed(4)}
                    </div>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
                    <div className="text-[10px] text-slate-400 font-sans">Quant MSE</div>
                    <div className="font-bold text-slate-800 dark:text-slate-200">
                      {metrics.quantization_distortion.mse.toFixed(4)}
                    </div>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
                    <div className="text-[10px] text-slate-400 font-sans">Quant ||E||_∞</div>
                    <div className="font-bold text-slate-800 dark:text-slate-200">
                      {metrics.quantization_distortion.linf.toFixed(4)}
                    </div>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
                    <div className="text-[10px] text-slate-400 font-sans">Quant Std</div>
                    <div className="font-bold text-slate-800 dark:text-slate-200">
                      {metrics.quantization_distortion.std.toFixed(4)}
                    </div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
                  <div className="text-[10px] text-slate-500 dark:text-slate-400 font-medium">
                    Changed Spatial Pixels (τ = {metrics.l0_threshold ?? l0Threshold})
                  </div>
                  <div className="text-sm font-bold font-mono text-slate-800 dark:text-slate-200">
                    {metrics.changed_pixels_count?.toLocaleString() ?? metrics.modified_pixels_count.toLocaleString()}{' '}
                    <span className="text-xs text-slate-400 font-normal">
                      ({metrics.changed_pixels_percent?.toFixed(1) ?? metrics.modified_pixels_percent.toFixed(1)}%)
                    </span>
                  </div>
                </div>
                <div className="p-2 rounded bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
                  <div className="text-[10px] text-slate-500 dark:text-slate-400 font-medium">
                    Changed Scalar Elements (H×W×C)
                  </div>
                  <div className="text-sm font-bold font-mono text-slate-800 dark:text-slate-200">
                    {metrics.changed_elements_count?.toLocaleString() ?? '—'}{' '}
                    <span className="text-xs text-slate-400 font-normal">
                      ({metrics.changed_elements_percent?.toFixed(1) ?? '—'}%)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Multi-Step Local Pixel & Neighborhood Submatrix Inspector */}
      {pi && (
        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100 dark:border-slate-800">
            <div className="flex items-center gap-2">
              <Crosshair className="w-4 h-4 text-sky-500" />
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Multi-Step Pixel Inspector at ({selectedPoint[0]}, {selectedPoint[1]})
              </h2>
            </div>

            {/* Coordinate Inputs */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">Coordinate:</span>
              <div className="flex items-center gap-1">
                <span className="text-xs text-slate-400 font-mono">x=</span>
                <input
                  type="number"
                  min="0"
                  max={metadata.width - 1}
                  value={selectedPoint[0]}
                  onChange={(e) =>
                    setSelectedPoint([
                      Math.max(0, Math.min(metadata.width - 1, parseInt(e.target.value) || 0)),
                      selectedPoint[1],
                    ])
                  }
                  className="w-16 px-2 py-1 text-xs font-mono rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200"
                />
              </div>
              <div className="flex items-center gap-1">
                <span className="text-xs text-slate-400 font-mono">y=</span>
                <input
                  type="number"
                  min="0"
                  max={metadata.height - 1}
                  value={selectedPoint[1]}
                  onChange={(e) =>
                    setSelectedPoint([
                      selectedPoint[0],
                      Math.max(0, Math.min(metadata.height - 1, parseInt(e.target.value) || 0)),
                    ])
                  }
                  className="w-16 px-2 py-1 text-xs font-mono rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200"
                />
              </div>

              {/* Submatrix Radius Selector */}
              <select
                value={submatrixRadius}
                onChange={(e) => setSubmatrixRadius(parseInt(e.target.value))}
                className="ml-2 px-2 py-1 text-xs rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200"
              >
                <option value={1}>3×3 Window (r=1)</option>
                <option value={2}>5×5 Window (r=2)</option>
                <option value={3}>7×7 Window (r=3)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Step-by-Step Transition Table */}
            <div className="border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden">
              <div className="px-3 py-2 bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-800 text-xs font-semibold text-slate-700 dark:text-slate-300">
                Step-by-Step Pixel Value Transition
              </div>
              <table className="w-full text-xs">
                <thead className="bg-slate-100/60 dark:bg-slate-800/60 text-slate-600 dark:text-slate-400">
                  <tr>
                    <th className="px-3 py-1.5 text-left font-semibold">Stage</th>
                    <th className="px-3 py-1.5 text-center font-semibold">RGB [R, G, B]</th>
                    <th className="px-3 py-1.5 text-right font-semibold">Δ (Step)</th>
                    <th className="px-3 py-1.5 text-right font-semibold">Δ (vs Base)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  {/* Base Original */}
                  <tr className="bg-slate-50/50 dark:bg-slate-900/50 font-medium">
                    <td className="px-3 py-2 font-semibold text-sky-700 dark:text-sky-300">0. Base Image</td>
                    <td className="px-3 py-2 text-center font-mono">
                      [{pi.original_rgb.join(', ')}]
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-400">0.0</td>
                    <td className="px-3 py-2 text-right font-mono text-slate-400">0.0</td>
                  </tr>

                  {/* Intermediate Steps */}
                  {(pi.intermediate_pixels || []).map((stepPix) => (
                    <tr key={stepPix.step_index}>
                      <td className="px-3 py-2 font-medium text-slate-700 dark:text-slate-300">
                        {stepPix.step_index}. {stepPix.step_name}
                      </td>
                      <td className="px-3 py-2 text-center font-mono font-bold">
                        [{stepPix.rgb.join(', ')}]
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-slate-600 dark:text-slate-300">
                        [{stepPix.delta_from_previous.map((d) => (d > 0 ? `+${d}` : d)).join(', ')}]
                      </td>
                      <td className="px-3 py-2 text-right font-mono font-semibold text-amber-600 dark:text-amber-400">
                        [{stepPix.delta_from_original.map((d) => (d > 0 ? `+${d}` : d)).join(', ')}]
                      </td>
                    </tr>
                  ))}

                  {/* Final Perturbed */}
                  <tr className="bg-amber-50/50 dark:bg-amber-950/40 font-bold">
                    <td className="px-3 py-2 text-amber-800 dark:text-amber-300">Final Result</td>
                    <td className="px-3 py-2 text-center font-mono text-amber-800 dark:text-amber-200">
                      [{pi.perturbed_rgb.join(', ')}]
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-500">—</td>
                    <td className="px-3 py-2 text-right font-mono text-amber-600 dark:text-amber-400">
                      [{pi.delta_rgb.map((d) => (d > 0 ? `+${d}` : d)).join(', ')}]
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Neighborhood Matrix Subgrid */}
            <div className="border border-slate-200 dark:border-slate-800 rounded-lg p-3 bg-slate-50/50 dark:bg-slate-900/50 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-700 dark:text-slate-300">
                  Total Delta Submatrix Window [ymin:{pi.submatrix_bounds.ymin}..ymax:{pi.submatrix_bounds.ymax}]
                </span>
                <div className="flex gap-1 text-[10px]">
                  {(['R', 'G', 'B', 'GRAY'] as const).map((ch) => (
                    <button
                      key={ch}
                      type="button"
                      onClick={() => setSubmatrixChannel(ch)}
                      className={`px-2 py-0.5 rounded font-bold transition-all ${
                        submatrixChannel === ch
                          ? 'bg-amber-600 text-white'
                          : 'bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      {ch}
                    </button>
                  ))}
                </div>
              </div>

              {/* Submatrix Grid Display */}
              <div className="overflow-x-auto">
                {(() => {
                  const matrix =
                    submatrixChannel === 'R'
                      ? pi.submatrix_delta_r
                      : submatrixChannel === 'G'
                      ? pi.submatrix_delta_g
                      : submatrixChannel === 'B'
                      ? pi.submatrix_delta_b
                      : pi.submatrix_delta_gray;

                  return (
                    <table className="mx-auto border-collapse text-[10px] font-mono">
                      <tbody>
                        {matrix.map((row, rIdx) => (
                          <tr key={rIdx}>
                            {row.map((val, cIdx) => {
                              const isCenter =
                                rIdx === submatrixRadius && cIdx === submatrixRadius;
                              const valNum = typeof val === 'number' ? val : 0;
                              return (
                                <td
                                  key={cIdx}
                                  className={`p-1 text-center border border-slate-200 dark:border-slate-800 min-w-[34px] ${
                                    isCenter
                                      ? 'ring-2 ring-amber-500 font-bold bg-amber-100 dark:bg-amber-950/60'
                                      : ''
                                  } ${
                                    valNum > 0
                                      ? 'text-emerald-700 dark:text-emerald-300'
                                      : valNum < 0
                                      ? 'text-rose-700 dark:text-rose-300'
                                      : 'text-slate-400'
                                  }`}
                                >
                                  {valNum > 0 ? `+${valNum.toFixed(0)}` : valNum.toFixed(0)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  );
                })()}
              </div>
              <div className="text-[10px] text-slate-400 text-center font-mono">
                Highlighted cell: center inspected coordinate ({selectedPoint[0]}, {selectedPoint[1]})
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
