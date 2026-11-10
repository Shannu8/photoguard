"""Mathematical utilities for image analysis and linear algebra operations.

Numeric conventions
-------------------
* Stored images:   uint8  ndarray, shape (H, W, 3), range [0, 255].
* Grayscale:       float64 ndarray, shape (H, W),   range [0.0, 255.0].
* Perturbation δ:  float64 ndarray, same shape as image, range [-255, +255].
* Normalized imgs: float64 ndarray, same spatial shape, range [0.0, 1.0].
  Use normalize_to_01() / denormalize_from_01() to convert.

L2 norm scope
-------------
All L2 norms in this module are Euclidean / Frobenius norms computed across
the **entire flattened tensor** (H × W × C elements), not per-sample or
per-channel, unless a function name or docstring states otherwise.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
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


def calculate_mae(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Calculates Mean Absolute Error (MAE):
    MAE = (1 / N) * sum(|original - perturbed|)
    """
    if original.shape != perturbed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs perturbed {perturbed.shape}")
    diff = np.abs(original.astype(np.float64) - perturbed.astype(np.float64))
    return float(np.mean(diff))


def calculate_norms_and_diff_stats(
    delta: np.ndarray, l0_threshold: float = 1e-4
) -> Dict[str, Any]:
    """
    Computes mathematical matrix norms and statistical bounds for perturbation tensor Δ = I' - I.
    Distinguishes changed scalar elements from changed spatial pixel coordinates.
    Supports configurable L0 threshold τ and per-channel decomposition.
    """
    d_float = delta.astype(np.float64)
    d_flat = d_float.flatten()
    abs_d = np.abs(d_flat)

    total_elements = int(len(d_flat))
    changed_elements = int(np.sum(abs_d > l0_threshold))
    changed_elements_pct = float(
        (changed_elements / total_elements * 100.0) if total_elements > 0 else 0.0
    )

    # Spatial pixel calculation (H x W)
    if d_float.ndim == 3 and d_float.shape[2] >= 3:
        # Pixel (y, x) is changed if any color channel exceeds threshold
        pixel_max_abs = np.max(np.abs(d_float[:, :, :3]), axis=2)
        total_pixels = int(d_float.shape[0] * d_float.shape[1])
        changed_pixels = int(np.sum(pixel_max_abs > l0_threshold))
    else:
        total_pixels = total_elements
        changed_pixels = changed_elements

    changed_pixels_pct = float(
        (changed_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
    )

    l1 = float(np.sum(abs_d))
    l2 = float(np.sqrt(np.sum(d_flat ** 2)))
    linf = float(np.max(abs_d)) if len(abs_d) > 0 else 0.0
    mae = float(np.mean(abs_d)) if len(abs_d) > 0 else 0.0
    mse = float(np.mean(d_flat ** 2)) if len(d_flat) > 0 else 0.0
    mean_diff = float(np.mean(d_flat)) if len(d_flat) > 0 else 0.0
    std_diff = float(np.std(d_flat)) if len(d_flat) > 0 else 0.0
    min_diff = float(np.min(d_flat)) if len(d_flat) > 0 else 0.0
    max_diff = float(np.max(d_flat)) if len(d_flat) > 0 else 0.0

    # Per-channel statistics breakdown
    per_channel: Dict[str, Dict[str, float]] = {}
    if d_float.ndim == 3 and d_float.shape[2] >= 3:
        channel_names = ["R", "G", "B"]
        for c_idx, c_name in enumerate(channel_names):
            c_data = d_float[:, :, c_idx].flatten()
            c_abs = np.abs(c_data)
            per_channel[c_name] = {
                "l1_norm": float(np.sum(c_abs)),
                "l2_norm": float(np.sqrt(np.sum(c_data ** 2))),
                "linf_norm": float(np.max(c_abs)) if len(c_abs) > 0 else 0.0,
                "mae": float(np.mean(c_abs)) if len(c_abs) > 0 else 0.0,
                "mse": float(np.mean(c_data ** 2)) if len(c_data) > 0 else 0.0,
                "mean_diff": float(np.mean(c_data)) if len(c_data) > 0 else 0.0,
                "std_diff": float(np.std(c_data)) if len(c_data) > 0 else 0.0,
                "min_diff": float(np.min(c_data)) if len(c_data) > 0 else 0.0,
                "max_diff": float(np.max(c_data)) if len(c_data) > 0 else 0.0,
                "changed_count": int(np.sum(c_abs > l0_threshold)),
            }

    return {
        "l1_norm": l1,
        "l2_norm": l2,
        "linf_norm": linf,
        "mae": mae,
        "mse": mse,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "min_diff": min_diff,
        "max_diff": max_diff,
        "max_abs_diff": linf,
        "l0_threshold": float(l0_threshold),
        "changed_elements_count": changed_elements,
        "total_elements_count": total_elements,
        "changed_elements_percent": round(changed_elements_pct, 2),
        "changed_pixels_count": changed_pixels,
        "total_pixels_count": total_pixels,
        "changed_pixels_percent": round(changed_pixels_pct, 2),
        # Backward-compatible aliases
        "modified_pixels_count": changed_pixels,
        "modified_pixels_percent": round(changed_pixels_pct, 2),
        "per_channel": per_channel,
    }


def calculate_rmse(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Calculates Root Mean Squared Error (RMSE) between original and perturbed arrays:
    RMSE = sqrt(MSE) = sqrt((1/N) * sum((original - perturbed)^2))

    Args:
        original:  Reference array (any shape).
        perturbed: Perturbed array (must match original shape).

    Returns:
        float: RMSE value ≥ 0.  Identical arrays → 0.0.
    """
    return float(np.sqrt(calculate_mse(original, perturbed)))


def calculate_quantization_metrics(float_tensor: np.ndarray) -> Dict[str, float]:
    """
    Computes quantization distortion introduced by rounding a float64 tensor to uint8.

    This measures the information loss from converting continuous perturbation values
    (or any float image tensor in [0, 255]) to discrete 8-bit integers.

    The input is expected to be float64 in the range [0, 255].  Values outside this
    range are clipped before quantisation (consistent with uint8 storage).

    Metrics returned:
        quant_mae   – Mean Absolute Error between float and quantized tensor.
        quant_mse   – Mean Squared Error between float and quantized tensor.
        quant_rmse  – Root MSE (sqrt of quant_mse).
        quant_max_error – Maximum absolute deviation (L∞ of quantization error).
        quant_psnr  – PSNR of the quantized tensor relative to the float tensor,
                      with MAX = 255.  Returns 100.0 when quant_mse ≈ 0.

    Args:
        float_tensor: float64 ndarray, any shape.  Interpreted as [0, 255] intensity.

    Returns:
        dict with keys: quant_mae, quant_mse, quant_rmse, quant_max_error, quant_psnr.
    """
    f = np.asarray(float_tensor, dtype=np.float64)
    # Clip to valid uint8 range before quantization
    clipped = np.clip(f, 0.0, 255.0)
    quantized = np.round(clipped).astype(np.uint8).astype(np.float64)

    diff = clipped - quantized
    abs_diff = np.abs(diff)
    n = diff.size

    quant_mae = float(np.mean(abs_diff)) if n > 0 else 0.0
    quant_mse = float(np.mean(diff ** 2)) if n > 0 else 0.0
    quant_rmse = float(np.sqrt(quant_mse))
    quant_max_error = float(np.max(abs_diff)) if n > 0 else 0.0
    quant_psnr = (
        100.0
        if quant_mse <= 1e-12
        else float(10.0 * np.log10((255.0 ** 2) / quant_mse))
    )

    return {
        "quant_mae": quant_mae,
        "quant_mse": quant_mse,
        "quant_rmse": quant_rmse,
        "quant_max_error": quant_max_error,
        "quant_psnr": quant_psnr,
    }


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """
    Converts an image array from [0, 255] integer/float range to normalized float64 [0, 1].

    Formula:  normalized = arr / 255.0

    This is an *explicit* unit conversion — do not apply to arrays that are already
    normalized, perturbation deltas, or any data whose range is not [0, 255].

    Args:
        arr: ndarray of any shape.  Typically uint8 or float64 in [0, 255].

    Returns:
        float64 ndarray of the same shape, values in [0.0, 1.0].
        Values outside [0, 255] on input produce values outside [0, 1] on output;
        no clipping is applied here — use np.clip explicitly if needed.
    """
    return np.asarray(arr, dtype=np.float64) / 255.0


def denormalize_from_01(arr: np.ndarray) -> np.ndarray:
    """
    Converts a normalized float64 [0, 1] array back to uint8 [0, 255].

    Formula:  uint8_out = clip(round(arr * 255.0), 0, 255).astype(uint8)

    Clipping and rounding are applied because the target type is uint8.
    Do not use this function to convert perturbation deltas (which live in
    [-255, +255] and are not normalized in the same sense).

    Args:
        arr: float64 ndarray, expected range [0, 1].

    Returns:
        uint8 ndarray of the same shape, values in [0, 255].
    """
    scaled = np.asarray(arr, dtype=np.float64) * 255.0
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def image_to_normalized_float(img: np.ndarray) -> np.ndarray:
    """
    Explicitly converts an 8-bit image array to a normalized float64 tensor in [0.0, 1.0].
    Preserves grayscale and RGB structure and dimensions. Does not mutate the input array in place.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(img).__name__}")
    return img.astype(np.float64, copy=True) / 255.0


def normalized_float_to_image(tensor: np.ndarray) -> np.ndarray:
    """
    Explicitly quantizes a normalized float64 [0.0, 1.0] tensor to an 8-bit uint8 image in [0, 255].
    Applies mathematical rounding and clipping to [0, 255].
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(tensor).__name__}")
    scaled = tensor.astype(np.float64, copy=True) * 255.0
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def legacy_float_to_normalized(arr: np.ndarray, is_delta: bool = False) -> np.ndarray:
    """
    Explicit compatibility adapter:
    Converts legacy [0, 255] float image data (or [-255, 255] delta) to normalized float64 [0, 1] (or [-1, 1]).
    Requires explicit `is_delta` parameter. Does not guess range by inspecting values.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(arr).__name__}")
    return arr.astype(np.float64, copy=True) / 255.0


def normalized_to_legacy_float(arr: np.ndarray, is_delta: bool = False) -> np.ndarray:
    """
    Explicit compatibility adapter:
    Converts normalized float64 data [0, 1] (or [-1, 1] delta) to legacy float64 [0, 255] (or [-255, 255]).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(arr).__name__}")
    return arr.astype(np.float64, copy=True) * 255.0


def calculate_quantization_distortion(
    continuous_tensor: np.ndarray,
    quantized_tensor: np.ndarray,
) -> Dict[str, float]:
    """
    Computes quantization distortion metrics between continuous floating-point tensor
    and quantized discrete tensor.
    """
    if continuous_tensor.shape != quantized_tensor.shape:
        raise ValueError(
            f"Shape mismatch: continuous {continuous_tensor.shape} vs quantized {quantized_tensor.shape}"
        )
    diff = continuous_tensor.astype(np.float64) - quantized_tensor.astype(np.float64)
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff ** 2)) if diff.size > 0 else 0.0
    mae = float(np.mean(abs_diff)) if diff.size > 0 else 0.0
    rmse = float(np.sqrt(mse))
    max_err = float(np.max(abs_diff)) if diff.size > 0 else 0.0
    std_err = float(np.std(diff)) if diff.size > 0 else 0.0
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_error": max_err,
        "std": std_err,
    }


def calculate_lpips_hook(
    original: np.ndarray, perturbed: np.ndarray
) -> Optional[float]:
    """
    Placeholder/hook for Phase 5 Learned Perceptual Image Patch Similarity (LPIPS).
    In Phase 5, this hook integrates deep feature perceptual loss (e.g. AlexNet/VGG).
    Currently returns None — LPIPS is reserved for Phase 5 (requires lpips library).
    """
    return None


def calculate_ssim(original: np.ndarray, perturbed: np.ndarray, max_val: float = 255.0) -> float:
    """
    Calculates Structural Similarity Index Measure (SSIM) between original and perturbed image.
    Uses the standard Wang et al. (2004) formulation with Gaussian weighting (sigma=1.5).
    For 3D RGB arrays, computes mean SSIM across channels.
    """
    if original.shape != perturbed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs perturbed {perturbed.shape}")

    orig_f = original.astype(np.float64)
    pert_f = perturbed.astype(np.float64)

    def _ssim_2d(img1: np.ndarray, img2: np.ndarray) -> float:
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        sigma = 1.5

        mu1 = gaussian_filter(img1, sigma=sigma, mode="reflect")
        mu2 = gaussian_filter(img2, sigma=sigma, mode="reflect")

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = gaussian_filter(img1 * img1, sigma=sigma, mode="reflect") - mu1_sq
        sigma2_sq = gaussian_filter(img2 * img2, sigma=sigma, mode="reflect") - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, sigma=sigma, mode="reflect") - mu1_mu2

        ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return float(np.mean(ssim_map))

    if orig_f.ndim == 2:
        return _ssim_2d(orig_f, pert_f)
    elif orig_f.ndim == 3:
        channels = [
            _ssim_2d(orig_f[:, :, c], pert_f[:, :, c])
            for c in range(min(orig_f.shape[2], 3))
        ]
        return float(np.mean(channels))
    else:
        raise ValueError(f"Unsupported array dimension: {orig_f.ndim}")



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

