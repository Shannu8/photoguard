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

