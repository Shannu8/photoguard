/**
 * API client communicating with FastAPI backend.
 */

import type {
  ImageMetadata,
  PixelInspection,
  ImageStatistics,
  MatrixRankData,
  EigenAnalysisData,
  SVDAnalysisData,
  ReconstructionData,
} from '../types';

const BASE_URL = '/api';

export async function uploadImage(file: File): Promise<ImageMetadata> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(errorData.detail || 'Upload failed');
  }

  return res.json();
}

export async function loadSampleImage(): Promise<ImageMetadata> {
  const res = await fetch(`${BASE_URL}/sample-image`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Failed to load sample image' }));
    throw new Error(errorData.detail || 'Failed to load sample image');
  }
  return res.json();
}

export async function getImageMetadata(imageId: string): Promise<ImageMetadata> {
  const res = await fetch(`${BASE_URL}/images/${imageId}`);
  if (!res.ok) {
    throw new Error('Image not found');
  }
  return res.json();
}

export async function inspectPixel(
  imageId: string,
  x: number,
  y: number,
  radius: number = 3
): Promise<PixelInspection> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/inspect-pixel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x, y, submatrix_radius: radius }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Inspection failed' }));
    throw new Error(err.detail || 'Inspection failed');
  }
  return res.json();
}

export async function getImageStatistics(imageId: string): Promise<ImageStatistics> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/statistics`);
  if (!res.ok) {
    throw new Error('Failed to fetch statistics');
  }
  return res.json();
}

export async function getMatrixRank(imageId: string): Promise<MatrixRankData> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/matrix-rank`);
  if (!res.ok) {
    throw new Error('Failed to fetch matrix rank');
  }
  return res.json();
}

export async function getEigenAnalysis(
  imageId: string,
  matrixType: 'gram_row' | 'gram_col' | 'covariance' | 'patch' = 'gram_row',
  patchX?: number,
  patchY?: number,
  patchSize: number = 8
): Promise<EigenAnalysisData> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/eigen-analysis`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      matrix_type: matrixType,
      patch_x: patchX,
      patch_y: patchY,
      patch_size: patchSize,
      max_components: 50,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Eigen analysis failed' }));
    throw new Error(err.detail || 'Eigen analysis failed');
  }
  return res.json();
}

export async function getSVDAnalysis(imageId: string): Promise<SVDAnalysisData> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/svd-analysis`);
  if (!res.ok) {
    throw new Error('Failed to fetch SVD analysis');
  }
  return res.json();
}

export async function reconstructImage(
  imageId: string,
  k: number,
  mode: 'rgb' | 'grayscale' = 'rgb'
): Promise<ReconstructionData> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/reconstruct`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ k, mode }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Reconstruction failed' }));
    throw new Error(err.detail || 'Reconstruction failed');
  }
  return res.json();
}
