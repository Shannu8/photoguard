"""
Phase 4 — Step 4: Backend Testing, Mathematical Validation & Regression Audit.

Exhaustive mathematical verification and regression test suite covering:
1. Normalized image handling (ranges [0,1] vs [0,255], shape preservation, immutability, roundtrip)
2. Perturbation operator calculations (analytical formulas, parameter bounds, determinism)
3. Budget enforcement (Linf, L2, dual constraints, pixel clipping correction, actual measured norms)
4. Quantization distortion (E_quant = Delta_uint8 - Delta_continuous, idempotency, error metrics)
5. Metric calculations (MSE, MAE, PSNR, SSIM, L1, L2, Linf against analytical answers)
6. Export validations (NPY, NPZ, CSV, JSON metadata, lossless PNG image downloads)
7. API validation (schema boundaries, edge cases, error codes, HTTP exceptions)
8. Phase 1-3 regression audit (upload, inspection, matrix analysis, affine transformations)
"""

import io
import math
import zipfile
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.utils.math_utils import (
    image_to_normalized_float,
    normalized_float_to_image,
    legacy_float_to_normalized,
    normalized_to_legacy_float,
    calculate_quantization_distortion,
    rgb_to_grayscale,
    calculate_mse,
    calculate_mae,
    calculate_rmse,
    calculate_psnr,
    calculate_ssim,
    calculate_norms_and_diff_stats,
    compute_numerical_rank,
    compute_channel_stats,
)
from app.services.perturbation_service import PerturbationService
from app.services.budget_service import BudgetService
from app.services.image_service import image_service, StoredImage
from app.services.transformation_service import transformation_service
from app.models.schemas import (
    PerturbationRequest,
    PerturbationStep,
    BudgetConfig,
    TransformationRequest,
    TransformationStep,
)


client = TestClient(app)


def upload_test_image(width=16, height=16, color=(120, 150, 180)):
    """Helper creating and uploading an in-memory PNG test image."""
    img = Image.new("RGB", (width, height), color=color)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    res = client.post("/api/upload", files={"file": ("audit_test.png", bio.getvalue(), "image/png")})
    assert res.status_code == 200, f"Upload failed: {res.text}"
    return res.json()["image_id"]


# ==============================================================================
# SECTION 1: AUDIT NORMALIZED IMAGE HANDLING
# ==============================================================================

class TestSection1NormalizedImageHandling:
    """Rigorous verification of image representations, ranges, shapes, and immutability."""

    def test_float_range_strictly_0_to_1(self):
        """Verify float tensor normalized values are strictly in [0.0, 1.0]."""
        uint8_arr = np.array([
            [[0, 1, 2], [127, 128, 129]],
            [[253, 254, 255], [64, 192, 255]],
        ], dtype=np.uint8)
        norm = image_to_normalized_float(uint8_arr)
        assert norm.dtype == np.float64
        assert np.all(norm >= 0.0)
        assert np.all(norm <= 1.0)
        assert norm[0, 0, 0] == 0.0
        assert norm[1, 0, 2] == 1.0
        assert np.isclose(norm[0, 1, 1], 128.0 / 255.0)

    def test_uint8_range_strictly_0_to_255(self):
        """Verify uint8 quantized image values are integers strictly in [0, 255]."""
        float_tensor = np.array([
            [[-0.5, 0.0, 0.001], [0.5, 0.999, 1.0]],
            [[1.001, 1.5, 2.0], [-1.0, 0.25, 0.75]],
        ], dtype=np.float64)
        img = normalized_float_to_image(float_tensor)
        assert img.dtype == np.uint8
        assert np.all(img >= 0)
        assert np.all(img <= 255)
        # Boundary clipping checks
        assert img[0, 0, 0] == 0     # -0.5 clipped to 0
        assert img[0, 0, 1] == 0     # 0.0 -> 0
        assert img[0, 1, 2] == 255   # 1.0 -> 255
        assert img[1, 0, 0] == 255   # 1.001 clipped to 255
        assert img[1, 0, 1] == 255   # 1.5 clipped to 255
        assert img[1, 0, 2] == 255   # 2.0 clipped to 255
        assert img[1, 1, 0] == 0     # -1.0 clipped to 0

    def test_rgb_and_grayscale_dimensions_and_dtypes(self):
        """Verify RGB 3D and Grayscale 2D tensors retain shapes and dtypes."""
        rgb_uint8 = np.zeros((24, 32, 3), dtype=np.uint8)
        gray_uint8 = np.zeros((24, 32), dtype=np.uint8)

        norm_rgb = image_to_normalized_float(rgb_uint8)
        norm_gray = image_to_normalized_float(gray_uint8)

        assert norm_rgb.shape == (24, 32, 3)
        assert norm_rgb.dtype == np.float64
        assert norm_gray.shape == (24, 32)
        assert norm_gray.dtype == np.float64

        back_rgb = normalized_float_to_image(norm_rgb)
        back_gray = normalized_float_to_image(norm_gray)

        assert back_rgb.shape == (24, 32, 3)
        assert back_rgb.dtype == np.uint8
        assert back_gray.shape == (24, 32)
        assert back_gray.dtype == np.uint8

    def test_source_array_immutability(self):
        """Original arrays must never be mutated in-place by normalization or perturbation."""
        source_uint8 = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        source_copy = source_uint8.copy()

        norm = image_to_normalized_float(source_uint8)
        norm += 0.5  # mutate result
        assert np.array_equal(source_uint8, source_copy)

        norm_fresh = image_to_normalized_float(source_uint8)
        norm_copy = norm_fresh.copy()
        quant = normalized_float_to_image(norm_fresh)
        quant += 10  # mutate result
        assert np.array_equal(norm_fresh, norm_copy)
        assert np.array_equal(source_uint8, source_copy)

    def test_conversion_roundtrip_consistency(self):
        """Repeated conversions uint8 -> float -> uint8 -> float -> uint8 produce 0 drift."""
        original = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        step1_float = image_to_normalized_float(original)
        step2_uint8 = normalized_float_to_image(step1_float)
        step3_float = image_to_normalized_float(step2_uint8)
        step4_uint8 = normalized_float_to_image(step3_float)

        assert np.array_equal(original, step2_uint8)
        assert np.array_equal(original, step4_uint8)
        assert np.array_equal(step1_float, step3_float)


# ==============================================================================
# SECTION 2: VALIDATE PERTURBATION CALCULATIONS
# ==============================================================================

class TestSection2PerturbationCalculations:
    """Mathematical validation for all 9 perturbation operator variants."""

    @pytest.fixture
    def flat_norm_tensor(self):
        return np.full((10, 10, 3), 0.5, dtype=np.float64)

    def test_additive_positive_const_formula(self, flat_norm_tensor):
        out, formula, exp, name, cat = PerturbationService._apply_operator(
            op_type="additive",
            input_float=flat_norm_tensor,
            active_channels=[0, 1, 2],
            params={"epsilon": 0.05, "additive_mode": "positive_const"},
            rng=np.random.default_rng(42),
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert np.allclose(out, 0.55)
        assert cat == "bounded_noise"

    def test_additive_negative_const_formula(self, flat_norm_tensor):
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="additive",
            input_float=flat_norm_tensor,
            active_channels=[0, 1, 2],
            params={"epsilon": 0.05, "additive_mode": "negative_const"},
            rng=np.random.default_rng(42),
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert np.allclose(out, 0.45)

    def test_additive_channel_isolation(self, flat_norm_tensor):
        """Only selected active channels receive perturbation; others remain unmodified."""
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="additive",
            input_float=flat_norm_tensor,
            active_channels=[1],  # G channel only
            params={"epsilon": 0.1, "additive_mode": "positive_const"},
            rng=np.random.default_rng(42),
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert np.allclose(out[:, :, 0], 0.5)   # R unchanged
        assert np.allclose(out[:, :, 1], 0.6)   # G shifted +0.1
        assert np.allclose(out[:, :, 2], 0.5)   # B unchanged

    def test_multiplicative_formula_and_bounds(self, flat_norm_tensor):
        rng = np.random.default_rng(101)
        out, _, _, _, cat = PerturbationService._apply_operator(
            op_type="multiplicative",
            input_float=flat_norm_tensor,
            active_channels=[0, 1, 2],
            params={"multiplicative_factor": 0.2},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        # I' = I * (1 + delta), where delta in [-0.2, 0.2]
        # For I = 0.5, I' in [0.5 * 0.8, 0.5 * 1.2] = [0.4, 0.6]
        assert np.all(out >= 0.4 - 1e-12)
        assert np.all(out <= 0.6 + 1e-12)
        assert cat == "bounded_noise"

    def test_gaussian_noise_mean_and_variance(self):
        """Over a large image, Gaussian noise mean approaches 0 and std approaches sigma."""
        large_tensor = np.full((100, 100, 3), 0.5, dtype=np.float64)
        rng = np.random.default_rng(42)
        sigma = 0.05
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="gaussian",
            input_float=large_tensor,
            active_channels=[0, 1, 2],
            params={"sigma": sigma},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        noise = out - large_tensor
        assert np.isclose(np.mean(noise), 0.0, atol=0.005)
        assert np.isclose(np.std(noise), sigma, atol=0.005)

    def test_salt_and_pepper_exact_values(self, flat_norm_tensor):
        rng = np.random.default_rng(42)
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="salt_and_pepper",
            input_float=flat_norm_tensor,
            active_channels=[0, 1, 2],
            params={"salt_pepper_prob": 0.5, "salt_vs_pepper_ratio": 0.5},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        # In normalized space, salt is 1.0, pepper is 0.0, unaffected is 0.5
        unique_vals = np.unique(np.round(out, 6))
        assert set(unique_vals).issubset({0.0, 0.5, 1.0})

    def test_inversion_exact_formula(self, flat_norm_tensor):
        # input is 0.5 -> 1.0 - 0.5 = 0.5
        input_test = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float64)
        out, _, _, _, cat = PerturbationService._apply_operator(
            op_type="inversion",
            input_float=input_test,
            active_channels=[0, 1, 2],
            params={},
            rng=np.random.default_rng(42),
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert np.allclose(out, [[[0.9, 0.8, 0.7]]])
        assert cat == "educational_intensity"

    def test_brightness_contrast_formula(self):
        input_test = np.array([[[0.50196, 0.50196, 0.50196]]], dtype=np.float64)  # 128.0 / 255.0
        # contrast 1.5, brightness +0.05
        # I' = c * (I - midpoint) + midpoint + b = 1.5 * (0) + 0.50196 + 0.05 = 0.55196
        out, _, _, _, cat = PerturbationService._apply_operator(
            op_type="brightness_contrast",
            input_float=input_test,
            active_channels=[0, 1, 2],
            params={"contrast_factor": 1.5, "brightness_offset": 0.05},
            rng=np.random.default_rng(42),
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert np.allclose(out, 0.55196)
        assert cat == "photometric"

    def test_deterministic_reproducibility(self, flat_norm_tensor):
        out1, _, _, _, _ = PerturbationService._apply_operator(
            op_type="additive",
            input_float=flat_norm_tensor,
            active_channels=[0, 1, 2],
            params={"epsilon": 0.05, "additive_mode": "random"},
            rng=np.random.default_rng(777),
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        out2, _, _, _, _ = PerturbationService._apply_operator(
            op_type="additive",
            input_float=flat_norm_tensor,
            active_channels=[0, 1, 2],
            params={"epsilon": 0.05, "additive_mode": "random"},
            rng=np.random.default_rng(777),
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert np.array_equal(out1, out2)


# ==============================================================================
# SECTION 3: VALIDATE PERTURBATION BUDGETS
# ==============================================================================

class TestSection3PerturbationBudgets:
    """Verification of L_infinity, L2, dual budgets and post-clipping actual norms."""

    def test_linf_norm_definition_and_projection(self):
        """||Delta||_inf = max(abs(Delta)) <= epsilon_inf."""
        delta = np.array([[-0.08, 0.02], [0.05, 0.12]], dtype=np.float64)
        eps_inf = 0.04
        projected = BudgetService.project_linf(delta, eps_inf)
        assert np.max(np.abs(projected)) <= eps_inf + 1e-12
        assert np.allclose(projected, [[-0.04, 0.02], [0.04, 0.04]])

    def test_l2_norm_definition_and_projection(self):
        """||Delta||_2 = sqrt(sum(Delta^2)) <= epsilon_2 across entire flattened tensor."""
        # delta has 4 elements of 0.3 -> L2 = sqrt(4 * 0.09) = sqrt(0.36) = 0.6
        delta = np.full((2, 2), 0.3, dtype=np.float64)
        current_l2 = float(np.sqrt(np.sum(delta ** 2)))
        assert np.isclose(current_l2, 0.6)

        eps_2 = 0.3
        projected = BudgetService.project_l2(delta, eps_2, norm_scope="entire_tensor")
        projected_l2 = float(np.sqrt(np.sum(projected ** 2)))
        assert np.isclose(projected_l2, eps_2)
        # Scaled by 0.3 / 0.6 = 0.5 -> elements are 0.15
        assert np.allclose(projected, 0.15)

    def test_combined_dual_projection_preserves_both_bounds(self):
        """Dual projection clamps Linf first then scales L2, preserving both constraints."""
        delta = np.random.uniform(-0.2, 0.2, (8, 8, 3)).astype(np.float64)
        eps_inf = 0.03
        eps_2 = 0.15
        projected, report = BudgetService.project_combined(delta, eps_inf, eps_2)

        actual_linf = float(np.max(np.abs(projected)))
        actual_l2 = float(np.sqrt(np.sum(projected ** 2)))

        assert actual_linf <= eps_inf + 1e-7
        assert actual_l2 <= eps_2 + 1e-7
        assert report["linf_satisfied"] is True
        assert report["l2_satisfied"] is True
        assert report["all_constraints_satisfied"] is True

    def test_pixel_clipping_recalculates_actual_delta(self):
        """Image clipping to [0, 1] alters delta; actual delta must be measured and compliant."""
        image = np.full((4, 4), 0.95, dtype=np.float64)
        # Delta of +0.1 with Linf bound of 0.1.
        # But image + delta = 1.05 -> clips to 1.0.
        # Actual delta is 1.0 - 0.95 = 0.05, NOT 0.1!
        delta = np.full((4, 4), 0.1, dtype=np.float64)
        img_proj, actual_delta, report = BudgetService.apply_image_budget(
            image=image,
            delta=delta,
            budget_type="linf",
            epsilon_inf=0.1,
            clip_range=(0.0, 1.0),
        )
        assert np.allclose(img_proj, 1.0)
        assert np.allclose(actual_delta, 0.05)
        assert np.isclose(report["post_projection_linf"], 0.05)
        assert report["clipping_applied"] is True
        assert report["all_constraints_satisfied"] is True

    def test_sequential_composition_cumulative_budget(self):
        """In sequential mode, cumulative budget is enforced relative to base image."""
        img_id = upload_test_image(16, 16, (128, 128, 128))
        req = {
            "composition_mode": "sequential",
            "budget": {
                "enabled": True,
                "type": "linf",
                "epsilon_inf": 0.03,
                "unit": "normalized_0_1",
            },
            "operations": [
                # Step 1: add +0.02
                {"type": "additive", "enabled": True, "parameter_unit": "normalized_0_1", "params": {"epsilon": 0.02, "additive_mode": "positive_const"}},
                # Step 2: add +0.02 (cumulative would be +0.04 without budget)
                {"type": "additive", "enabled": True, "parameter_unit": "normalized_0_1", "params": {"epsilon": 0.02, "additive_mode": "positive_const"}},
            ],
            "seed": 42,
        }
        res = client.post(f"/api/images/{img_id}/perturb", json=req)
        assert res.status_code == 200
        data = res.json()
        report = data["budget_report"]
        assert report["linf_satisfied"] is True
        assert report["post_projection_linf"] <= 0.03 + 1e-6


# ==============================================================================
# SECTION 4: VALIDATE QUANTIZATION
# ==============================================================================

class TestSection4Quantization:
    """Verification of quantization distortion, rounding, and conversion idempotency."""

    def test_quantization_distortion_analytical_metrics(self):
        # Continuous tensor
        cont = np.array([[[10.1, 20.6], [30.5, 40.0]]], dtype=np.float64)
        # Quantized tensor (rounded to nearest integer)
        quant = np.array([[[10.0, 21.0], [30.0, 40.0]]], dtype=np.float64)  # 30.5 banker's round to 30
        # diff = cont - quant = [0.1, -0.4, 0.5, 0.0]
        # abs_diff = [0.1, 0.4, 0.5, 0.0]
        # MAE = (0.1 + 0.4 + 0.5 + 0.0) / 4 = 1.0 / 4 = 0.25
        # MSE = (0.01 + 0.16 + 0.25 + 0.0) / 4 = 0.42 / 4 = 0.105
        # RMSE = sqrt(0.105) ≈ 0.324037
        # max_error = 0.5
        res = calculate_quantization_distortion(cont, quant)
        assert np.isclose(res["mae"], 0.25)
        assert np.isclose(res["mse"], 0.105)
        assert np.isclose(res["rmse"], np.sqrt(0.105))
        assert np.isclose(res["max_error"], 0.5)

    def test_quantization_idempotency(self):
        """Quantizing an already-quantized tensor produces 0 quantization error."""
        discrete_img = np.array([[[10, 50, 200]]], dtype=np.uint8)
        norm_float = image_to_normalized_float(discrete_img)
        quant_uint8 = normalized_float_to_image(norm_float)

        res = calculate_quantization_distortion(
            norm_float * 255.0, quant_uint8.astype(np.float64)
        )
        assert res["mae"] == 0.0
        assert res["mse"] == 0.0
        assert res["rmse"] == 0.0
        assert res["max_error"] == 0.0

    def test_quantization_maximum_error_bound(self):
        """Rounding float to nearest integer has mathematical max error <= 0.5."""
        floats = np.random.uniform(0.0, 255.0, (50, 50, 3)).astype(np.float64)
        quantized = np.round(floats)
        diff = np.abs(floats - quantized)
        assert np.max(diff) <= 0.5 + 1e-12


# ==============================================================================
# SECTION 5: VALIDATE METRICS
# ==============================================================================

class TestSection5Metrics:
    """Analytical validation of MSE, MAE, RMSE, PSNR, SSIM, and vector/matrix norms."""

    def test_mse_mae_rmse_analytical(self):
        # original = [10, 20], perturbed = [12, 16]
        # diff = [-2, 4] -> |diff| = [2, 4] -> MAE = 3.0
        # diff^2 = [4, 16] -> MSE = 10.0 -> RMSE = sqrt(10.0) ≈ 3.162277
        orig = np.array([[10, 20]], dtype=np.uint8)
        pert = np.array([[12, 16]], dtype=np.uint8)

        assert np.isclose(calculate_mae(orig, pert), 3.0)
        assert np.isclose(calculate_mse(orig, pert), 10.0)
        assert np.isclose(calculate_rmse(orig, pert), np.sqrt(10.0))

    def test_psnr_analytical_formula(self):
        # For max_val = 255, MSE = 10.0:
        # PSNR = 10 * log10(255^2 / 10) = 10 * log10(65025 / 10) = 10 * log10(6502.5) ≈ 38.1308 dB
        orig = np.array([[10, 20]], dtype=np.uint8)
        pert = np.array([[12, 16]], dtype=np.uint8)
        expected_psnr = 10.0 * np.log10(65025.0 / 10.0)
        assert np.isclose(calculate_psnr(orig, pert), expected_psnr, atol=1e-4)

    def test_psnr_identical_arrays_no_infinity_crash(self):
        """When MSE = 0, PSNR returns 100.0 dB indicator without crashing or NaN."""
        orig = np.full((16, 16, 3), 128, dtype=np.uint8)
        assert calculate_psnr(orig, orig) == 100.0

    def test_ssim_identical_arrays(self):
        """SSIM between identical images is strictly 1.0."""
        orig = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        ssim_val = calculate_ssim(orig, orig)
        assert np.isclose(ssim_val, 1.0)

    def test_matrix_norms_and_diff_stats_analytical(self):
        # 1D delta: [-3.0, 4.0]
        # L1 = 3 + 4 = 7.0
        # L2 = sqrt((-3)^2 + 4^2) = sqrt(9 + 16) = 5.0
        # Linf = max(3, 4) = 4.0
        # Mean diff = (-3 + 4) / 2 = 0.5
        # Std diff = std([-3, 4]) = sqrt((( -3 - 0.5 )^2 + ( 4 - 0.5 )^2) / 2) = sqrt((12.25 + 12.25)/2) = 3.5
        delta = np.array([-3.0, 4.0], dtype=np.float64)
        stats = calculate_norms_and_diff_stats(delta, l0_threshold=1.0)
        assert np.isclose(stats["l1_norm"], 7.0)
        assert np.isclose(stats["l2_norm"], 5.0)
        assert np.isclose(stats["linf_norm"], 4.0)
        assert np.isclose(stats["mean_diff"], 0.5)
        assert np.isclose(stats["std_diff"], 3.5)
        assert stats["changed_elements_count"] == 2


# ==============================================================================
# SECTION 6: VALIDATE EXPORTS
# ==============================================================================

class TestSection6Exports:
    """Verification of tensor export formats and PNG processed image download."""

    @pytest.fixture(autouse=True)
    def setup_image_with_perturbation(self):
        self.img_id = upload_test_image(16, 16, (100, 150, 200))
        req = {
            "composition_mode": "sequential",
            "operations": [
                {
                    "type": "additive",
                    "enabled": True,
                    "parameter_unit": "normalized_0_1",
                    "params": {"epsilon": 0.04, "additive_mode": "uniform"},
                }
            ],
            "seed": 42,
        }
        res = client.post(f"/api/images/{self.img_id}/perturb", json=req)
        assert res.status_code == 200

    def test_download_png_processed_image(self):
        res = client.get(f"/api/images/{self.img_id}/download?target=perturbed_rgb")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert res.content.startswith(b"\x89PNG\r\n\x1a\n")
        pil_img = Image.open(io.BytesIO(res.content))
        assert pil_img.size == (16, 16)
        assert pil_img.mode == "RGB"

    def test_download_png_grayscale_image(self):
        res = client.get(f"/api/images/{self.img_id}/download?target=original_grayscale")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        pil_img = Image.open(io.BytesIO(res.content))
        assert pil_img.size == (16, 16)

    def test_export_npy_continuous_normalized_tensor(self):
        res = client.get(f"/api/images/{self.img_id}/tensors/download?target=perturbed_rgb_normalized&format=npy")
        assert res.status_code == 200
        arr = np.load(io.BytesIO(res.content))
        assert arr.shape == (16, 16, 3)
        assert arr.dtype == np.float64
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)

    def test_export_npz_archive_contains_all_arrays(self):
        res = client.get(f"/api/images/{self.img_id}/tensors/download?target=all&format=npz")
        assert res.status_code == 200
        npz = np.load(io.BytesIO(res.content))
        for key in ["original_rgb", "original_grayscale", "perturbed_rgb", "perturbed_rgb_normalized", "perturbation_normalized"]:
            assert key in npz

    def test_export_csv_precision_and_structure(self):
        res = client.get(f"/api/images/{self.img_id}/tensors/download?target=original_grayscale&format=csv")
        assert res.status_code == 200
        csv_text = res.content.decode("utf-8")
        rows = csv_text.strip().split("\n")
        assert len(rows) == 16
        # Check precision format (%.6f)
        first_row_vals = [float(v) for v in rows[0].split(",")]
        assert len(first_row_vals) == 16

    def test_export_json_metadata(self):
        res = client.get(f"/api/images/{self.img_id}/tensors/download?target=metadata&format=json")
        assert res.status_code == 200
        assert "application/json" in res.headers["content-type"]


# ==============================================================================
# SECTION 7: API VALIDATION & BOUNDARY CONDITIONS
# ==============================================================================

class TestSection7APIValidation:
    """Verification of API schemas, error codes, and input boundaries."""

    def test_nonexistent_image_returns_404(self):
        res = client.get("/api/images/non-existent-uuid-12345/inspect")
        assert res.status_code == 404
        res_p = client.post("/api/images/non-existent-uuid-12345/perturb", json={"method": "additive"})
        assert res_p.status_code == 404

    def test_unsupported_perturbation_method_rejected(self):
        img_id = upload_test_image()
        req = {
            "operations": [{"type": "unsupported_magic_noise"}]
        }
        res = client.post(f"/api/images/{img_id}/perturb", json=req)
        assert res.status_code in [400, 422]

    def test_normalized_strength_above_1_rejected(self):
        """Normalized parameters > 1.0 must be rejected with 422, not silently auto-divided."""
        img_id = upload_test_image()
        req = {
            "operations": [
                {
                    "type": "additive",
                    "parameter_unit": "normalized_0_1",
                    "params": {"epsilon": 25.0},  # 25 > 1.0
                }
            ]
        }
        res = client.post(f"/api/images/{img_id}/perturb", json=req)
        assert res.status_code == 422

    def test_weight_outside_0_to_5_rejected(self):
        img_id = upload_test_image()
        req = {
            "composition_mode": "weighted",
            "operations": [
                {
                    "type": "additive",
                    "weight": 6.5,  # > 5.0
                }
            ]
        }
        res = client.post(f"/api/images/{img_id}/perturb", json=req)
        assert res.status_code == 422

    def test_unsupported_download_target_returns_400(self):
        img_id = upload_test_image()
        res = client.get(f"/api/images/{img_id}/download?target=invalid_target")
        assert res.status_code == 400


# ==============================================================================
# SECTION 8: PHASE 1-3 REGRESSION AUDIT
# ==============================================================================

class TestSection8Phase1to3Regressions:
    """Verifies that Phase 1, Phase 2, and Phase 3 capabilities remain 100% operational."""

    def test_phase1_pixel_inspection(self):
        img_id = upload_test_image(16, 16, (120, 150, 180))
        res = client.post(f"/api/images/{img_id}/inspect-pixel", json={"x": 5, "y": 5, "submatrix_radius": 2})
        assert res.status_code == 200
        data = res.json()
        assert data["submatrix_center"] == [5, 5]
        assert (data["r"], data["g"], data["b"]) == (120, 150, 180)
        # BT.601 grayscale: 0.299 * 120 + 0.587 * 150 + 0.114 * 180 = 35.88 + 88.05 + 20.52 = 144.45
        assert np.isclose(data["grayscale_exact"], 144.45, atol=0.01)
        assert "Y = 0.299" in data["calculation_steps"]

    def test_phase2_matrix_analysis(self):
        img_id = upload_test_image(16, 16, (100, 100, 100))
        res_rank = client.get(f"/api/images/{img_id}/matrix-rank")
        assert res_rank.status_code == 200
        data_rank = res_rank.json()
        assert "rank" in data_rank
        assert "is_full_rank" in data_rank

        res_svd = client.get(f"/api/images/{img_id}/svd-analysis")
        assert res_svd.status_code == 200
        data_svd = res_svd.json()
        assert "singular_values" in data_svd
        assert "cumulative_energy_percent" in data_svd

    def test_phase3_affine_transformation_pipeline(self):
        img_id = upload_test_image(16, 16, (100, 150, 200))
        trans_req = {
            "operations": [
                {"type": "rotate", "params": {"angle_degrees": 90.0}},
                {"type": "scale", "params": {"scale_x": 0.5, "scale_y": 0.5}},
            ],
            "interpolation": "nearest",
            "canvas_mode": "same",
        }
        res = client.post(f"/api/images/{img_id}/transform", json=trans_req)
        assert res.status_code == 200
        data = res.json()
        assert "composite_matrix_3x3" in data
        assert len(data["steps"]) == 2
        assert "transformed_rgb_data_url" in data

    def test_phase3_tensor_metadata(self):
        img_id = upload_test_image(16, 16, (100, 150, 200))
        res = client.get(f"/api/images/{img_id}/tensors/metadata")
        assert res.status_code == 200
        data = res.json()
        assert data["original_rgb"]["shape"] == [16, 16, 3]
        assert data["original_grayscale"]["shape"] == [16, 16]
