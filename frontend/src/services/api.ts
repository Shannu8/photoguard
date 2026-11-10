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
  TransformationStep,
  TransformationResponse,
  TensorMetadataResponse,
  PerturbationRequest,
  PerturbationResponse,
} from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const BASE_URL = `${API_BASE}/api`;

/**
 * Handles HTTP responses, extracting clear diagnostic error details.
 */
async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) {
    return res.json();
  }

  let errorMessage = '';

  // Attempt to parse JSON error response from FastAPI
  try {
    const errorData = await res.json();
    if (typeof errorData.detail === 'string') {
      errorMessage = errorData.detail;
    } else if (Array.isArray(errorData.detail)) {
      // Pydantic/FastAPI validation error list
      errorMessage = errorData.detail.map((e: any) => e.msg || e.message).join('; ');
    }
  } catch {
    // Response body is not JSON (e.g. proxy HTML 502/504 error)
  }

  // Provide meaningful fallback messages if no detail was parsed
  if (!errorMessage) {
    switch (res.status) {
      case 400:
        errorMessage = 'Invalid request. Please verify the uploaded file format and integrity.';
        break;
      case 404:
        errorMessage = 'Resource not found. Please ensure the backend API route is registered.';
        break;
      case 413:
        errorMessage = 'File exceeds size limit: Maximum file size allowed is 20 MB.';
        break;
      case 502:
      case 503:
      case 504:
        errorMessage = 'Backend unavailable. Could not establish connection to the FastAPI server at http://127.0.0.1:8000.';
        break;
      case 500:
        errorMessage = 'Server processing error occurred while analyzing the image matrix.';
        break;
      default:
        errorMessage = `Request failed with status ${res.status} (${res.statusText || 'Error'}).`;
        break;
    }
  }

  throw new Error(errorMessage);
}

/**
 * Checks whether the backend service is reachable.
 */
export async function checkBackendHealth(): Promise<{ online: boolean; message?: string }> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { method: 'GET' });
    if (res.ok) {
      return { online: true };
    }
    return { online: false, message: `Backend responded with HTTP status ${res.status}` };
  } catch (err: any) {
    return {
      online: false,
      message: 'Backend server is not reachable on port 8000.',
    };
  }
}

export async function uploadImage(file: File): Promise<ImageMetadata> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    return await handleResponse<ImageMetadata>(res);
  } catch (err: any) {
    // Detect browser network exceptions (e.g. Failed to fetch / Connection Refused)
    if (err instanceof TypeError && (err.message.includes('fetch') || err.message.includes('Network'))) {
      throw new Error(
        'Backend unavailable: Unable to connect to http://127.0.0.1:8000. Please ensure the backend server is running.'
      );
    }
    throw err;
  }
}

export async function loadSampleImage(
  preset: 'harmonic' | 'low_rank_blocks' | 'texture_noise' = 'harmonic'
): Promise<ImageMetadata> {
  try {
    const res = await fetch(`${BASE_URL}/sample-image?preset=${preset}`);
    return await handleResponse<ImageMetadata>(res);
  } catch (err: any) {
    if (err instanceof TypeError && (err.message.includes('fetch') || err.message.includes('Network'))) {
      throw new Error('Backend unavailable: Unable to load sample image.');
    }
    throw err;
  }
}

export async function getImageMetadata(imageId: string): Promise<ImageMetadata> {
  const res = await fetch(`${BASE_URL}/images/${imageId}`);
  return handleResponse<ImageMetadata>(res);
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
  return handleResponse<PixelInspection>(res);
}

export async function getImageStatistics(imageId: string): Promise<ImageStatistics> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/statistics`);
  return handleResponse<ImageStatistics>(res);
}

export async function getMatrixRank(imageId: string): Promise<MatrixRankData> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/matrix-rank`);
  return handleResponse<MatrixRankData>(res);
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
  return handleResponse<EigenAnalysisData>(res);
}

export async function getSVDAnalysis(imageId: string): Promise<SVDAnalysisData> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/svd-analysis`);
  return handleResponse<SVDAnalysisData>(res);
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
  return handleResponse<ReconstructionData>(res);
}

export async function applyTransformations(
  imageId: string,
  operations: TransformationStep[],
  interpolation: 'bilinear' | 'nearest' = 'bilinear',
  canvasMode: 'fit' | 'same' = 'fit',
  selectedPoint?: [number, number] | null
): Promise<TransformationResponse> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/transform`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      operations: operations.map((op) => ({ type: op.type, params: op.params })),
      interpolation,
      canvas_mode: canvasMode,
      selected_point: selectedPoint || null,
    }),
  });
  return handleResponse<TransformationResponse>(res);
}

export async function getTensorMetadata(imageId: string): Promise<TensorMetadataResponse> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/tensors/metadata`);
  return handleResponse<TensorMetadataResponse>(res);
}

export function getTensorDownloadUrl(
  imageId: string,
  target:
    | 'original_rgb'
    | 'transformed_rgb'
    | 'original_grayscale'
    | 'transformed_grayscale'
    | 'perturbed_rgb'
    | 'perturbed_grayscale'
    | 'perturbation'
    | 'difference'
    | 'absolute_difference'
    | 'all'
    | string,
  format: 'npy' | 'npz' | 'csv' | 'json' | string
): string {
  return `${BASE_URL}/images/${imageId}/tensors/download?target=${target}&format=${format}`;
}

export async function applyPerturbation(
  imageId: string,
  req: PerturbationRequest
): Promise<PerturbationResponse> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/perturb`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return handleResponse<PerturbationResponse>(res);
}

export async function getPerturbationMetadata(imageId: string): Promise<any> {
  const res = await fetch(`${BASE_URL}/images/${imageId}/perturbation/metadata`);
  return handleResponse<any>(res);
}

