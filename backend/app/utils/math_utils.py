"""Mathematical utilities for image analysis and linear algebra operations."""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def rgb_to_grayscale(rgb_array: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image array (H, W, 3) to grayscale (H, W)
    using the ITU-R BT.601 standard formula:
    Y = 0.299 * R + 0.587 * G + 0.114 * B

    Args:
        rgb_array: NumPy array of shape (H, W, 3) or (H, W, 4) with RGB channels.

    Returns:
        np.ndarray: Grayscale 2D array of shape (H, W) as float64.
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
        raise ValueError(f"Expected RGB image array with 3 channels, got shape {rgb_array.shape}")

    # Use float64 for mathematical precision
    r = rgb_array[:, :, 0].astype(np.float64)
    g = rgb_array[:, :, 1].astype(np.float64)
    b = rgb_array[:, :, 2].astype(np.float64)

    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale


def format_pixel_calculation(r: float, g: float, b: float) -> Tuple[float, str]:
    """
    Computes grayscale value and a human-readable transparent calculation string.
    """
    y = 0.299 * r + 0.587 * g + 0.114 * b
    calc_str = (
        f"Y = 0.299 * ({r:.1f}) + 0.587 * ({g:.1f}) + 0.114 * ({b:.1f})\n"
        f"  = {0.299 * r:.3f} + {0.587 * g:.3f} + {0.114 * b:.3f}\n"
        f"  = {y:.3f}"
    )
    return y, calc_str


def calculate_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE) between original and reconstructed arrays:
    MSE = (1 / N) * sum((original - reconstructed)^2)
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        )
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    return mse


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB):
    PSNR = 10 * log10(MAX^2 / MSE) = 20 * log10(MAX / sqrt(MSE))

    If MSE is 0, returns infinity (represented as float('inf') or 100.0 dB for practical displays).
    """
    mse = calculate_mse(original, reconstructed)
    if mse <= 1e-12:
        return 100.0  # Perfect reconstruction indicator
    psnr = float(10.0 * np.log10((max_val ** 2) / mse))
    return psnr


def compute_numerical_rank(matrix: np.ndarray, custom_tol: Optional[float] = None) -> Tuple[int, float, bool, np.ndarray]:
    """
    Computes numerical rank of a 2D matrix using Singular Value Decomposition.

    Args:
        matrix: 2D NumPy array (H, W).
        custom_tol: Optional user tolerance. If None, uses NumPy default:
                    tol = max(singular_values) * max(M, N) * eps

    Returns:
        rank: integer rank
        tol: effective tolerance threshold used
        is_full_rank: whether rank == min(H, W)
        singular_values: all singular values of the matrix
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix for rank calculation, got {matrix.ndim}D")

    s = np.linalg.svd(matrix, compute_uv=False)
    m, n = matrix.shape
    eps = np.finfo(matrix.dtype if np.issubdtype(matrix.dtype, np.floating) else np.float64).eps
    default_tol = float(s[0] * max(m, n) * eps) if len(s) > 0 and s[0] > 0 else 0.0

    tolerance = custom_tol if custom_tol is not None else default_tol
    rank = int(np.sum(s > tolerance))
    is_full_rank = (rank == min(m, n))

    return rank, tolerance, is_full_rank, s


def compute_channel_stats(data: np.ndarray) -> Dict[str, Any]:
    """
    Computes descriptive statistics and a 256-bin histogram for an intensity array.
    """
    flat = data.flatten().astype(np.float64)
    if len(flat) == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "histogram": [0] * 256,
            "bin_edges": list(range(257))
        }

    # Bins 0 through 256 for integer range [0, 255]
    hist, edges = np.histogram(flat, bins=256, range=(0, 256))

    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "histogram": hist.tolist(),
        "bin_edges": [float(e) for e in edges]
    }

