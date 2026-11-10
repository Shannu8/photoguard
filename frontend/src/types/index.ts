/**
 * TypeScript interface definitions matching backend Pydantic models.
 */

export interface ImageMetadata {
  image_id: string;
  filename: string;
  width: int;
  height: int;
  format: string;
  channels: int;
  total_pixels: int;
  color_mode: string;
  file_size_bytes: int;
  rgb_data_url: string;
  grayscale_data_url: string;
}

export type int = number;

export interface PixelInspection {
  x: int;
  y: int;
  r: int;
  g: int;
  b: int;
  grayscale_exact: number;
  grayscale_uint8: int;
  calculation_steps: string;
  submatrix_center: [int, int];
  submatrix_bounds: {
    row_start: int;
    row_end: int;
    col_start: int;
    col_end: int;
  };
  submatrix_r: int[][];
  submatrix_g: int[][];
  submatrix_b: int[][];
  submatrix_grayscale: number[][];
}

export interface ChannelStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
  histogram: int[];
  bin_edges: number[];
}

export interface ImageStatistics {
  grayscale: ChannelStats;
  red: ChannelStats;
  green: ChannelStats;
  blue: ChannelStats;
}

export interface MatrixRankData {
  rows: int;
  cols: int;
  rank: int;
  is_full_rank: boolean;
  numerical_tolerance: number;
  max_possible_rank: int;
  singular_values_top: number[];
  singular_values_bottom: number[];
  explanation: string;
  scientific_notes: string;
}

export interface EigenAnalysisData {
  matrix_type: string;
  matrix_dimensions: [int, int];
  matrix_description: string;
  total_eigenvalues_count: int;
  eigenvalues_real: number[];
  eigenvalues_imag: number[];
  has_complex: boolean;
  sample_matrix_preview: number[][];
  dominant_eigenvectors: number[][];
  eigenvalues_energy_distribution: number[];
  explanation: string;
  pca_distinction: string;
}

export interface SVDAnalysisData {
  rows: int;
  cols: int;
  total_components: int;
  singular_values: number[];
  cumulative_energy_percent: number[];
  condition_number: number;
  effective_rank: int;
  explanation: string;
  k_sample_points?: number[];
  theoretical_psnr_curve?: number[];
}

export interface ReconstructionData {
  k: int;
  total_components: int;
  retained_percentage: number;
  mode: string;
  mse: number;
  psnr: number;
  energy_preserved_percent: number;
  compression_ratio: number;
  original_data_url: string;
  reconstructed_data_url: string;
  difference_data_url: string;
  explanation: string;
}

export type TransformationStepType =
  | 'rotate'
  | 'scale'
  | 'reflect'
  | 'shear'
  | 'translate'
  | 'random_affine'
  | 'resize'
  | 'jpeg_compression';

export interface TransformationStep {
  id?: string;
  type: TransformationStepType;
  params: Record<string, any>;
}

export interface MatrixStepInfo {
  name: string;
  operation_type: string;
  matrix_3x3: number[][];
  formula_latex: string;
  description: string;
}

export interface TransformationResponse {
  original_width: int;
  original_height: int;
  transformed_width: int;
  transformed_height: int;
  composite_matrix_3x3: number[][];
  steps: MatrixStepInfo[];
  matrix_multiplication_explanation: string;
  original_point: [number, number] | null;
  transformed_point: [number, number] | null;
  transformed_rgb_data_url: string;
  transformed_grayscale_data_url: string;
  interpolation: string;
  canvas_mode: string;
  determinant: number;
  is_invertible: boolean;
}

export interface TensorInfo {
  name: string;
  shape: int[];
  dtype: string;
  total_elements: int;
  estimated_size_bytes: int;
}

export interface TensorMetadataResponse {
  has_transformed: boolean;
  original_rgb: TensorInfo;
  original_grayscale: TensorInfo;
  transformed_rgb: TensorInfo | null;
  transformed_grayscale: TensorInfo | null;
  has_perturbed?: boolean;
  perturbed_rgb?: TensorInfo | null;
  perturbed_grayscale?: TensorInfo | null;
  perturbation_delta?: TensorInfo | null;
}

// Phase 4: Mathematical Image Perturbation Engine Types
export type PerturbationMethod =
  | 'additive'
  | 'multiplicative'
  | 'gaussian'
  | 'salt_and_pepper'
  | 'inversion'
  | 'brightness_contrast';

export type CompositionMode = 'sequential' | 'weighted';

export interface BudgetConfig {
  enabled?: boolean;
  type: 'none' | 'linf' | 'l2' | 'both';
  epsilon_inf?: number | null;
  epsilon_linf?: number | null;
  epsilon_2?: number | null;
  epsilon_l2?: number | null;
  l0_threshold?: number | null;
  max_changed_pixel_ratio?: number | null;
  norm_scope?: 'entire_tensor' | 'per_channel' | 'per_pixel';
  unit?: 'normalized_0_1' | 'pixel_0_255';
}

export interface ChannelMetrics {
  channel: string;
  l1_norm: number;
  l2_norm: number;
  linf_norm: number;
  mae: number;
  mse: number;
  mean_diff: number;
  std_diff: number;
  min_diff: number;
  max_diff: number;
  changed_count: number;
}

export interface PerturbationStep {
  id: string;
  type: PerturbationMethod;
  enabled: boolean;
  weight: number; // 0.0 to 5.0
  strength?: number | null;
  parameter_unit?: 'normalized_0_1' | 'pixel_0_255';
  seed?: number | null;
  channels: ('R' | 'G' | 'B')[];
  params: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface PerturbationConfig {
  version: string;
  composition_mode: CompositionMode;
  base_source: 'original' | 'transformed';
  budget?: BudgetConfig | null;
  seed?: number | null;
  operations: PerturbationStep[];
  l0_threshold?: number;
  metadata?: Record<string, any>;
}

export interface IntermediatePixelValue {
  step_index: number;
  step_name: string;
  rgb: [number, number, number];
  delta_from_previous: [number, number, number];
  delta_from_original: [number, number, number];
}

export interface PerturbationStepResult {
  step_index: number;
  step_id: string;
  type: string;
  name: string;
  category: 'bounded_noise' | 'educational_intensity' | 'photometric';
  enabled: boolean;
  weight: number;
  formula_latex: string;
  preview_data_url: string;
  intermediate_metrics: PerturbationMetrics;
}

export interface PerturbationRequest {
  composition_mode?: CompositionMode;
  budget?: BudgetConfig | null;
  budget_unit?: 'normalized_0_1' | 'pixel_0_255';
  global_bound?: number | null;
  operations?: PerturbationStep[];
  l0_threshold?: number;
  method?: PerturbationMethod;
  base_source?: 'original' | 'transformed';
  channels?: ('R' | 'G' | 'B')[];
  epsilon?: number;
  additive_mode?: 'random' | 'uniform' | 'positive_const' | 'negative_const';
  multiplicative_factor?: number;
  sigma?: number;
  salt_pepper_prob?: number;
  salt_vs_pepper_ratio?: number;
  contrast_factor?: number;
  brightness_offset?: number;
  seed?: number | null;
  selected_point?: [number, number] | null;
  submatrix_radius?: number;
}

export interface PerturbationMetrics {
  mse: number;
  psnr: number;
  ssim: number;
  mae: number;
  l1_norm: number;
  l2_norm: number;
  linf_norm: number;
  max_abs_perturbation: number;
  min_difference: number;
  max_difference: number;
  mean_difference: number;
  std_diff?: number;
  l0_threshold?: number;
  changed_elements_count?: number;
  total_elements_count?: number;
  changed_elements_percent?: number;
  changed_pixels_count?: number;
  total_pixels_count: number;
  changed_pixels_percent?: number;
  modified_pixels_count: number;
  total_pixels_count_legacy?: number;
  modified_pixels_percent: number;
  lpips?: number | null;
  per_channel?: Record<string, ChannelMetrics>;
  quantization_distortion?: {
    mae: number;
    mse: number;
    linf: number;
    std: number;
  };
}

export interface PerturbationPixelInspection {
  x: number;
  y: number;
  original_rgb: [number, number, number];
  intermediate_pixels?: IntermediatePixelValue[];
  perturbed_rgb: [number, number, number];
  delta_rgb: [number, number, number];
  abs_delta_rgb: [number, number, number];
  original_gray: number;
  perturbed_gray: number;
  delta_gray: number;
  submatrix_radius: number;
  submatrix_bounds: {
    ymin: number;
    ymax: number;
    xmin: number;
    xmax: number;
  };
  submatrix_delta_r: number[][];
  submatrix_delta_g: number[][];
  submatrix_delta_b: number[][];
  submatrix_delta_gray: number[][];
}

export interface PerturbationResponse {
  image_id: string;
  method: string;
  composition_mode?: string;
  base_source: string;
  width: number;
  height: number;
  channels: number;
  global_bound?: number | null;
  budget_report?: Record<string, any> | null;
  parameters: Record<string, any>;
  reproducibility_seed: number | null;
  original_data_url: string;
  perturbed_rgb_data_url: string;
  perturbed_grayscale_data_url: string;
  difference_heatmap_data_url: string;
  signed_difference_data_url: string;
  metrics: PerturbationMetrics;
  pixel_inspection: PerturbationPixelInspection | null;
  intermediate_steps?: PerturbationStepResult[];
  method_explanation: string;
  mathematical_formula: string;
}


