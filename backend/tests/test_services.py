"""Unit tests verifying mathematical accuracy for Phase 1 & Phase 2 image analysis."""

import numpy as np
import pytest
from app.utils.math_utils import (
    rgb_to_grayscale,
    format_pixel_calculation,
    calculate_mse,
    calculate_psnr,
    compute_numerical_rank,
    compute_channel_stats,
)
from app.services.image_service import ImageService
from app.services.matrix_service import MatrixService
from app.services.eigen_service import EigenService
from app.services.svd_service import SVDService
from app.services.reconstruction_service import ReconstructionService
from app.models.schemas import EigenAnalysisRequest, ReconstructionRequest


# ==========================================
# 1. RGB to Grayscale Tests
# ==========================================

def test_rgb_to_grayscale_known_values():
    """Verify exact formula Y = 0.299R + 0.587G + 0.114B."""
    # 1x1 test pixels
    # Pure Red
    red_pixel = np.array([[[255, 0, 0]]], dtype=np.uint8)
    assert np.isclose(rgb_to_grayscale(red_pixel)[0, 0], 0.299 * 255, atol=1e-5)

    # Pure Green
    green_pixel = np.array([[[0, 255, 0]]], dtype=np.uint8)
    assert np.isclose(rgb_to_grayscale(green_pixel)[0, 0], 0.587 * 255, atol=1e-5)

    # Pure Blue
    blue_pixel = np.array([[[0, 0, 255]]], dtype=np.uint8)
    assert np.isclose(rgb_to_grayscale(blue_pixel)[0, 0], 0.114 * 255, atol=1e-5)

    # Pure White: 0.299 + 0.587 + 0.114 = 1.0 -> 255.0
    white_pixel = np.array([[[255, 255, 255]]], dtype=np.uint8)
    assert np.isclose(rgb_to_grayscale(white_pixel)[0, 0], 255.0, atol=1e-5)

    # Pure Black
    black_pixel = np.array([[[0, 0, 0]]], dtype=np.uint8)
    assert np.isclose(rgb_to_grayscale(black_pixel)[0, 0], 0.0, atol=1e-5)

    # Arbitrary known pixel [100, 150, 200]
    # Y = 0.299*100 + 0.587*150 + 0.114*200 = 29.9 + 88.05 + 22.8 = 140.75
    arb_pixel = np.array([[[100, 150, 200]]], dtype=np.uint8)
    expected_y = 0.299 * 100 + 0.587 * 150 + 0.114 * 200
    assert np.isclose(rgb_to_grayscale(arb_pixel)[0, 0], expected_y, atol=1e-5)


def test_pixel_calculation_formatting():
    """Verify transparent mathematical explanation string."""
    y, text = format_pixel_calculation(100, 150, 200)
    assert np.isclose(y, 140.75, atol=1e-5)
    assert "0.299" in text
    assert "0.587" in text
    assert "0.114" in text


# ==========================================
# 2. MSE & PSNR Metric Tests
# ==========================================

def test_mse_and_psnr():
    """Verify MSE and PSNR on exact numerical matrices."""
    a = np.array([[10.0, 20.0], [30.0, 40.0]])
    b = np.array([[10.0, 20.0], [30.0, 40.0]])

    # Identical arrays
    assert calculate_mse(a, b) == 0.0
    assert calculate_psnr(a, b) == 100.0

    # Perturbed by delta = 1.0 everywhere -> MSE = 1.0
    c = a + 1.0
    assert np.isclose(calculate_mse(a, c), 1.0, atol=1e-6)
    # PSNR = 20 * log10(255 / 1) = 48.1308 dB
    expected_psnr = 20.0 * np.log10(255.0 / 1.0)
    assert np.isclose(calculate_psnr(a, c), expected_psnr, atol=1e-4)


# ==========================================
# 3. Matrix Rank Tests
# ==========================================

def test_matrix_rank():
    """Verify rank calculation using known linear algebra matrices."""
    # Identity matrix of size 4x4 -> rank = 4 (full rank)
    i4 = np.eye(4)
    rank, tol, is_full, s = compute_numerical_rank(i4)
    assert rank == 4
    assert is_full is True
    assert len(s) == 4

    # All-ones matrix 4x4 -> rank = 1 (rank deficient)
    j4 = np.ones((4, 4))
    rank_j, _, is_full_j, _ = compute_numerical_rank(j4)
    assert rank_j == 1
    assert is_full_j is False

    # Outer product u @ v.T with non-zero vectors -> rank 1
    u = np.array([[1], [2], [3]])
    v = np.array([[4], [5], [6], [7]])
    outer = u @ v.T  # 3x4 matrix
    rank_outer, _, is_full_outer, _ = compute_numerical_rank(outer)
    assert rank_outer == 1
    assert is_full_outer is False

    # Matrix with linearly dependent rows (row3 = row1 + row2)
    m = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [5.0, 7.0, 9.0]
    ])
    rank_m, _, is_full_m, _ = compute_numerical_rank(m)
    assert rank_m == 2
    assert is_full_m is False


# ==========================================
# 4. Eigenvalue and Eigenvector Tests
# ==========================================

def test_eigenvalue_known_symmetric_matrix():
    """Verify Av = λv on known symmetric 2x2 matrix."""
    # Matrix [[2, 1], [1, 2]]: eigenvalues are 3 and 1
    mat = np.array([[2.0, 1.0], [1.0, 2.0]])
    vals, vecs = np.linalg.eig(mat)
    sorted_vals = np.sort(vals)[::-1]
    assert np.isclose(sorted_vals[0], 3.0, atol=1e-5)
    assert np.isclose(sorted_vals[1], 1.0, atol=1e-5)

    # Check Av = λv for each eigenvector
    for i in range(2):
        v = vecs[:, i]
        lam = vals[i]
        assert np.allclose(mat @ v, lam * v, atol=1e-5)


def test_complex_eigenvalues_non_symmetric():
    """Verify complex eigenvalue detection on 90-degree rotation matrix."""
    # [[0, -1], [1, 0]] has eigenvalues +i and -i
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    vals, _ = np.linalg.eig(rot)
    assert np.any(np.abs(np.imag(vals)) > 0.99)


# ==========================================
# 5. SVD & Eckart-Young Reconstruction Tests
# ==========================================

def test_svd_orthonormality_and_reconstruction():
    """Verify SVD decomposition U Σ V^T = A and low-rank approximation."""
    # Create arbitrary 6x4 rank-3 matrix
    np.random.seed(42)
    a = np.random.randn(6, 3) @ np.random.randn(3, 4)  # 6x4 matrix, min(6,4)=4
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    k = min(a.shape)  # 4

    # Check U orthonormal: U^T U = I_k
    assert np.allclose(u.T @ u, np.eye(k), atol=1e-5)
    # Check V orthonormal: V^T V = I (i.e. vt @ vt.T = I_k)
    assert np.allclose(vt @ vt.T, np.eye(k), atol=1e-5)
    # Check full reconstruction
    a_rec = (u * s) @ vt
    assert np.allclose(a, a_rec, atol=1e-5)

    # Check Eckart-Young property: rank-k approximation error
    # Rank-1 approximation
    a_k1 = (u[:, :1] * s[:1]) @ vt[:1, :]
    err_k1 = np.linalg.norm(a - a_k1, 'fro')

    # Rank-2 approximation
    a_k2 = (u[:, :2] * s[:2]) @ vt[:2, :]
    err_k2 = np.linalg.norm(a - a_k2, 'fro')

    # Error of rank-2 must be strictly less than error of rank-1
    assert err_k2 < err_k1


# ==========================================
# 6. Channel Statistics Tests
# ==========================================

def test_channel_statistics():
    """Verify min, max, mean, median, std, and 256-bin histogram."""
    data = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    stats = compute_channel_stats(data)

    assert stats["min"] == 10.0
    assert stats["max"] == 40.0
    assert stats["mean"] == 25.0
    assert stats["median"] == 25.0
    assert len(stats["histogram"]) == 256
    # 4 values total in histogram
    assert sum(stats["histogram"]) == 4
    assert stats["histogram"][10] == 1
    assert stats["histogram"][20] == 1
    assert stats["histogram"][30] == 1
    assert stats["histogram"][40] == 1
