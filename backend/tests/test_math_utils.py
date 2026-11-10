"""
Comprehensive unit tests for backend/app/utils/math_utils.py — Phase 4, Step 1.

Hand-calculated reference values are documented inline so reviewers can verify
mathematical correctness without running the code.

Conventions tested:
  - float64 inputs in [0, 255] (legacy image range)
  - float64 inputs in [0, 1]  (normalized range via normalize_to_01)
  - uint8 inputs
  - 2D (grayscale) and 3D (RGB) arrays
  - Zero delta, partial delta, full delta
  - Boundary conditions and shape mismatches
"""

import math
import numpy as np
import pytest

from backend.app.utils.math_utils import (
    calculate_mae,
    calculate_mse,
    calculate_psnr,
    calculate_rmse,
    calculate_ssim,
    calculate_norms_and_diff_stats,
    calculate_lpips_hook,
    calculate_quantization_metrics,
    normalize_to_01,
    denormalize_from_01,
    rgb_to_grayscale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb(h: int, w: int, fill: int = 0) -> np.ndarray:
    """Create a solid-colour uint8 RGB image."""
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _known_2x2_rgb():
    """
    2×2 RGB image with four distinct pixels.
    Layout (row, col): (0,0)=red, (0,1)=green, (1,0)=blue, (1,1)=white
    Values: [[255,0,0], [0,255,0], [0,0,255], [255,255,255]]
    """
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]
    img[0, 1] = [0, 255, 0]
    img[1, 0] = [0, 0, 255]
    img[1, 1] = [255, 255, 255]
    return img


# ---------------------------------------------------------------------------
# 1. calculate_mae
# ---------------------------------------------------------------------------

class TestMAE:
    def test_identical_arrays_zero(self):
        a = np.array([100.0, 200.0, 50.0])
        assert calculate_mae(a, a) == pytest.approx(0.0)

    def test_known_1d(self):
        # |10-7| + |20-15| + |30-32| = 3 + 5 + 2 = 10; mean = 10/3
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([7.0, 15.0, 32.0])
        assert calculate_mae(a, b) == pytest.approx(10.0 / 3.0, rel=1e-9)

    def test_rgb_image(self):
        orig = _make_rgb(4, 4, 100)
        pert = _make_rgb(4, 4, 110)
        assert calculate_mae(orig, pert) == pytest.approx(10.0)

    def test_uint8_vs_float_equivalent(self):
        orig_u = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        orig_f = orig_u.astype(np.float64)
        pert_u = np.array([[15, 25], [35, 45]], dtype=np.uint8)
        pert_f = pert_u.astype(np.float64)
        assert calculate_mae(orig_u, pert_u) == pytest.approx(calculate_mae(orig_f, pert_f))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_mae(np.zeros((3, 3)), np.zeros((4, 4)))

    def test_grayscale_2d(self):
        a = np.array([[0.0, 100.0], [200.0, 50.0]])
        b = np.array([[5.0, 90.0], [210.0, 40.0]])
        # |0-5|=5, |100-90|=10, |200-210|=10, |50-40|=10 → mean = 35/4 = 8.75
        assert calculate_mae(a, b) == pytest.approx(8.75)


# ---------------------------------------------------------------------------
# 2. calculate_mse
# ---------------------------------------------------------------------------

class TestMSE:
    def test_identical_zero(self):
        a = np.ones((5, 5, 3), dtype=np.float64) * 128.0
        assert calculate_mse(a, a) == pytest.approx(0.0)

    def test_known_scalar(self):
        # single element: (3.0 - 5.0)^2 = 4.0, mean = 4.0
        a = np.array([3.0])
        b = np.array([5.0])
        assert calculate_mse(a, b) == pytest.approx(4.0)

    def test_known_1d(self):
        # diffs: 1, 2, 3 → sq: 1, 4, 9 → mean = 14/3
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        assert calculate_mse(a, b) == pytest.approx(14.0 / 3.0, rel=1e-9)

    def test_rgb_uniform_offset(self):
        orig = _make_rgb(2, 2, 50)
        pert = _make_rgb(2, 2, 60)
        # all diffs = 10 → MSE = 100
        assert calculate_mse(orig, pert) == pytest.approx(100.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            calculate_mse(np.zeros((2, 2)), np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# 3. calculate_rmse
# ---------------------------------------------------------------------------

class TestRMSE:
    def test_identical_zero(self):
        a = np.ones((3, 3)) * 42.0
        assert calculate_rmse(a, a) == pytest.approx(0.0)

    def test_rmse_equals_sqrt_mse(self):
        """RMSE must always equal sqrt(MSE) for all inputs."""
        rng = np.random.default_rng(42)
        orig = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        pert = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        assert calculate_rmse(orig, pert) == pytest.approx(
            math.sqrt(calculate_mse(orig, pert)), rel=1e-9
        )

    def test_known_value(self):
        # uniform offset of 10 → MSE = 100 → RMSE = 10
        orig = _make_rgb(4, 4, 50)
        pert = _make_rgb(4, 4, 60)
        assert calculate_rmse(orig, pert) == pytest.approx(10.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            calculate_rmse(np.zeros((3,)), np.zeros((4,)))

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        a = rng.random((5, 5, 3))
        b = rng.random((5, 5, 3))
        assert calculate_rmse(a, b) >= 0.0


# ---------------------------------------------------------------------------
# 4. calculate_psnr
# ---------------------------------------------------------------------------

class TestPSNR:
    def test_identical_returns_100(self):
        a = np.full((4, 4, 3), 128, dtype=np.uint8)
        assert calculate_psnr(a, a) == pytest.approx(100.0)

    def test_known_psnr(self):
        # MSE = 100 (uniform +10 offset), MAX=255
        # PSNR = 10 * log10(255^2 / 100) = 10 * log10(650.25) ≈ 28.130...
        orig = _make_rgb(4, 4, 50)
        pert = _make_rgb(4, 4, 60)
        expected = 10.0 * math.log10(255.0 ** 2 / 100.0)
        assert calculate_psnr(orig, pert) == pytest.approx(expected, rel=1e-6)

    def test_max_val_01(self):
        """PSNR with max_val=1.0 for normalized images."""
        orig = np.array([0.5, 0.5])
        pert = np.array([0.6, 0.6])
        mse = 0.01
        expected = 10.0 * math.log10(1.0 / mse)
        assert calculate_psnr(orig, pert, max_val=1.0) == pytest.approx(expected, rel=1e-6)

    def test_psnr_decreases_with_more_distortion(self):
        orig = _make_rgb(8, 8, 128)
        pert_small = _make_rgb(8, 8, 129)
        pert_large = _make_rgb(8, 8, 150)
        assert calculate_psnr(orig, pert_small) > calculate_psnr(orig, pert_large)


# ---------------------------------------------------------------------------
# 5. calculate_norms_and_diff_stats
# ---------------------------------------------------------------------------

class TestNormsAndDiffStats:
    def _zero_delta(self):
        return np.zeros((4, 4, 3), dtype=np.float64)

    def test_zero_delta_all_zeros(self):
        stats = calculate_norms_and_diff_stats(self._zero_delta())
        for key in ("l1_norm", "l2_norm", "linf_norm", "mae", "mse", "mean_diff",
                    "std_diff", "changed_elements_count", "changed_pixels_count"):
            assert stats[key] == pytest.approx(0.0), f"{key} should be 0 for zero delta"

    def test_known_1d_delta(self):
        # delta = [3, -4] → L1=7, L2=5, Linf=4, MAE=3.5, MSE=12.5
        delta = np.array([3.0, -4.0])
        s = calculate_norms_and_diff_stats(delta)
        assert s["l1_norm"] == pytest.approx(7.0)
        assert s["l2_norm"] == pytest.approx(5.0)
        assert s["linf_norm"] == pytest.approx(4.0)
        assert s["mae"] == pytest.approx(3.5)
        assert s["mse"] == pytest.approx(12.5)

    def test_l0_threshold_configurable(self):
        # delta magnitudes: 0.001, 0.0, 5.0, 0.0005
        delta = np.array([0.001, 0.0, 5.0, -0.0005])
        # threshold=1e-4: 0.001 > 1e-4 ✓, 0.0 ✗, 5.0 ✓, 0.0005 > 1e-4 ✓ → 3 changed
        s_default = calculate_norms_and_diff_stats(delta, l0_threshold=1e-4)
        assert s_default["changed_elements_count"] == 3

        # threshold=0.01: 0.001 ✗, 5.0 ✓, 0.0005 ✗ → 1 changed
        s_strict = calculate_norms_and_diff_stats(delta, l0_threshold=0.01)
        assert s_strict["changed_elements_count"] == 1

    def test_per_channel_rgb(self):
        # R=+10, G=+0, B=-5 uniform across 2×2
        delta = np.zeros((2, 2, 3), dtype=np.float64)
        delta[:, :, 0] = 10.0
        delta[:, :, 2] = -5.0
        s = calculate_norms_and_diff_stats(delta)
        assert "R" in s["per_channel"]
        assert "G" in s["per_channel"]
        assert "B" in s["per_channel"]
        assert s["per_channel"]["R"]["mae"] == pytest.approx(10.0)
        assert s["per_channel"]["G"]["mae"] == pytest.approx(0.0)
        assert s["per_channel"]["B"]["mae"] == pytest.approx(5.0)

    def test_changed_pixels_vs_elements(self):
        """A pixel is changed if *any* channel exceeds threshold; check distinction."""
        delta = np.zeros((2, 2, 3), dtype=np.float64)
        # Only R channel of pixel (0,0) is changed
        delta[0, 0, 0] = 1.0
        s = calculate_norms_and_diff_stats(delta, l0_threshold=0.5)
        # 1 scalar element changed
        assert s["changed_elements_count"] == 1
        # 1 pixel has at least one channel changed
        assert s["changed_pixels_count"] == 1

    def test_std_diff_nonzero_for_varying_delta(self):
        delta = np.array([1.0, -1.0, 2.0, -2.0])
        s = calculate_norms_and_diff_stats(delta)
        assert s["std_diff"] > 0.0

    def test_backward_compat_aliases(self):
        delta = np.ones((3, 3, 3)) * 2.0
        s = calculate_norms_and_diff_stats(delta)
        assert s["modified_pixels_count"] == s["changed_pixels_count"]
        assert s["modified_pixels_percent"] == s["changed_pixels_percent"]
        assert s["max_abs_diff"] == s["linf_norm"]

    def test_total_counts_correct(self):
        h, w = 5, 7
        delta = np.ones((h, w, 3), dtype=np.float64)
        s = calculate_norms_and_diff_stats(delta)
        assert s["total_elements_count"] == h * w * 3
        assert s["total_pixels_count"] == h * w


# ---------------------------------------------------------------------------
# 6. calculate_quantization_metrics
# ---------------------------------------------------------------------------

class TestQuantizationMetrics:
    def test_integer_input_zero_error(self):
        """Already-integer float values → quantization error should be 0."""
        arr = np.array([0.0, 100.0, 200.0, 255.0])
        q = calculate_quantization_metrics(arr)
        assert q["quant_mae"] == pytest.approx(0.0)
        assert q["quant_mse"] == pytest.approx(0.0)
        assert q["quant_rmse"] == pytest.approx(0.0)
        assert q["quant_max_error"] == pytest.approx(0.0)
        assert q["quant_psnr"] == pytest.approx(100.0)

    def test_half_lsb_offset(self):
        """
        Each value is offset by exactly 0.4 (less than 0.5, rounds down).
        quant_mae = 0.4, quant_mse = 0.16
        """
        arr = np.array([10.4, 50.4, 100.4, 200.4])
        q = calculate_quantization_metrics(arr)
        assert q["quant_mae"] == pytest.approx(0.4, rel=1e-6)
        assert q["quant_mse"] == pytest.approx(0.16, rel=1e-6)
        assert q["quant_rmse"] == pytest.approx(0.4, rel=1e-6)
        assert q["quant_max_error"] == pytest.approx(0.4, rel=1e-5)

    def test_psnr_finite_for_nonzero_error(self):
        arr = np.array([10.3, 50.7, 100.2])
        q = calculate_quantization_metrics(arr)
        assert math.isfinite(q["quant_psnr"])
        assert q["quant_psnr"] > 0

    def test_out_of_range_clipped(self):
        """Values outside [0,255] are clipped before quantization."""
        arr = np.array([-10.0, 300.0])
        q = calculate_quantization_metrics(arr)
        # After clip: [0.0, 255.0] → both are integers → error = 0
        assert q["quant_mae"] == pytest.approx(0.0)

    def test_rgb_array_shape_preserved(self):
        arr = np.random.default_rng(99).uniform(0.0, 255.0, size=(4, 4, 3))
        q = calculate_quantization_metrics(arr)
        required_keys = {"quant_mae", "quant_mse", "quant_rmse", "quant_max_error", "quant_psnr"}
        assert required_keys == set(q.keys())

    def test_rmse_equals_sqrt_mse(self):
        arr = np.random.default_rng(13).uniform(0.0, 255.0, size=(10,))
        q = calculate_quantization_metrics(arr)
        assert q["quant_rmse"] == pytest.approx(math.sqrt(q["quant_mse"]), rel=1e-9)


# ---------------------------------------------------------------------------
# 7. normalize_to_01 / denormalize_from_01
# ---------------------------------------------------------------------------

class TestNormalizeDenormalize:
    def test_normalize_zero_and_max(self):
        arr = np.array([0.0, 255.0])
        n = normalize_to_01(arr)
        assert n[0] == pytest.approx(0.0)
        assert n[1] == pytest.approx(1.0)

    def test_normalize_midpoint(self):
        arr = np.array([127.5])
        n = normalize_to_01(arr)
        assert n[0] == pytest.approx(0.5)

    def test_normalize_dtype_float64(self):
        arr = np.array([100], dtype=np.uint8)
        n = normalize_to_01(arr)
        assert n.dtype == np.float64

    def test_denormalize_roundtrip(self):
        """uint8 → normalize → denormalize should recover the original uint8 values."""
        orig = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
        normalized = normalize_to_01(orig)
        recovered = denormalize_from_01(normalized)
        np.testing.assert_array_equal(orig, recovered)

    def test_denormalize_clips_overflow(self):
        """Values outside [0, 1] in normalized space are clipped to [0, 255]."""
        arr = np.array([-0.5, 0.5, 1.5])
        result = denormalize_from_01(arr)
        assert result[0] == 0
        assert result[2] == 255
        assert result.dtype == np.uint8

    def test_normalize_shape_preserved(self):
        arr = np.ones((3, 4, 3), dtype=np.uint8) * 128
        n = normalize_to_01(arr)
        assert n.shape == (3, 4, 3)

    def test_denormalize_dtype_uint8(self):
        arr = np.array([0.0, 0.5, 1.0])
        d = denormalize_from_01(arr)
        assert d.dtype == np.uint8

    def test_normalized_psnr_with_max_val_1(self):
        """
        PSNR computed on [0,1] data with max_val=1.0 should equal PSNR
        computed on [0,255] data with max_val=255.0 (scale-invariant).
        """
        orig = _make_rgb(4, 4, 100)
        pert = _make_rgb(4, 4, 110)
        psnr_255 = calculate_psnr(orig, pert, max_val=255.0)
        orig_n = normalize_to_01(orig)
        pert_n = normalize_to_01(pert)
        psnr_01 = calculate_psnr(orig_n, pert_n, max_val=1.0)
        assert psnr_255 == pytest.approx(psnr_01, rel=1e-6)


# ---------------------------------------------------------------------------
# 8. calculate_ssim
# ---------------------------------------------------------------------------

class TestSSIM:
    def test_identical_images_ssim_1(self):
        img = _make_rgb(16, 16, 128)
        ssim = calculate_ssim(img, img)
        # SSIM of identical images should be very close to 1.0
        assert ssim == pytest.approx(1.0, abs=1e-4)

    def test_ssim_range_0_to_1(self):
        orig = _make_rgb(32, 32, 100)
        pert = _make_rgb(32, 32, 200)
        ssim = calculate_ssim(orig, pert)
        assert 0.0 <= ssim <= 1.0

    def test_ssim_decreases_with_more_distortion(self):
        orig = _make_rgb(32, 32, 128)
        small = _make_rgb(32, 32, 130)
        large = _make_rgb(32, 32, 200)
        assert calculate_ssim(orig, small) > calculate_ssim(orig, large)

    def test_ssim_grayscale(self):
        g = np.ones((16, 16), dtype=np.float64) * 128.0
        assert calculate_ssim(g, g) == pytest.approx(1.0, abs=1e-4)

    def test_ssim_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            calculate_ssim(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))


# ---------------------------------------------------------------------------
# 9. calculate_lpips_hook
# ---------------------------------------------------------------------------

class TestLpipsHook:
    def test_returns_none(self):
        a = _make_rgb(8, 8, 50)
        b = _make_rgb(8, 8, 60)
        result = calculate_lpips_hook(a, b)
        assert result is None

    def test_returns_none_for_identical(self):
        a = _make_rgb(8, 8, 128)
        assert calculate_lpips_hook(a, a) is None


# ---------------------------------------------------------------------------
# 10. rgb_to_grayscale
# ---------------------------------------------------------------------------

class TestRgbToGrayscale:
    def test_pure_red(self):
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        img[0, 0] = [255, 0, 0]
        g = rgb_to_grayscale(img)
        assert g[0, 0] == pytest.approx(0.299 * 255, rel=1e-6)

    def test_pure_white(self):
        img = np.ones((2, 2, 3), dtype=np.uint8) * 255
        g = rgb_to_grayscale(img)
        np.testing.assert_allclose(g, 255.0, rtol=1e-6)

    def test_pure_black(self):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        g = rgb_to_grayscale(img)
        np.testing.assert_array_equal(g, 0.0)

    def test_output_shape(self):
        img = np.zeros((5, 7, 3), dtype=np.uint8)
        g = rgb_to_grayscale(img)
        assert g.shape == (5, 7)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            rgb_to_grayscale(np.zeros((5, 5)))


# ---------------------------------------------------------------------------
# 11. Cross-metric consistency
# ---------------------------------------------------------------------------

class TestCrossMetricConsistency:
    """Verify invariants that must hold across multiple metric functions."""

    def test_rmse_mae_ordering(self):
        """
        By Jensen's inequality RMSE ≥ MAE for any array.
        (RMSE = sqrt(E[x²]) ≥ E[|x|] = MAE when values are non-negative after abs.)
        """
        rng = np.random.default_rng(2024)
        orig = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        pert = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        assert calculate_rmse(orig, pert) >= calculate_mae(orig, pert) - 1e-9

    def test_norms_mse_matches_calculate_mse(self):
        """MSE from calculate_norms_and_diff_stats == calculate_mse for the same delta."""
        orig = _make_rgb(4, 4, 80)
        pert = _make_rgb(4, 4, 90)
        delta = pert.astype(np.float64) - orig.astype(np.float64)
        stats = calculate_norms_and_diff_stats(delta)
        # stats["mse"] is mean of delta^2 = MSE(orig, pert)
        assert stats["mse"] == pytest.approx(calculate_mse(orig, pert), rel=1e-9)

    def test_zero_perturbation_all_metrics_zero(self):
        orig = _make_rgb(6, 6, 150)
        pert = _make_rgb(6, 6, 150)
        assert calculate_mae(orig, pert) == pytest.approx(0.0)
        assert calculate_mse(orig, pert) == pytest.approx(0.0)
        assert calculate_rmse(orig, pert) == pytest.approx(0.0)
        assert calculate_psnr(orig, pert) == pytest.approx(100.0)
        # SSIM of identical uniform images ≈ 1.0
        assert calculate_ssim(orig, pert) == pytest.approx(1.0, abs=1e-4)
        delta = np.zeros_like(orig, dtype=np.float64)
        stats = calculate_norms_and_diff_stats(delta)
        assert stats["l2_norm"] == pytest.approx(0.0)
        assert stats["changed_pixels_count"] == 0
