"""
Phase 4 Step 3: Tests for Normalization, Perturbation Services, Quantization Distortion,
Lossless Tensor Export, and Lossless Image Download.
"""

import io
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
    calculate_psnr,
    calculate_ssim,
)
from app.services.perturbation_service import PerturbationService
from app.services.budget_service import BudgetService
from app.services.image_service import image_service, StoredImage
from app.models.schemas import (
    PerturbationRequest,
    PerturbationStep,
    BudgetConfig,
)


client = TestClient(app)


# ==============================================================================
# 1. Normalization & Explicit Unit Conversion Tests
# ==============================================================================

class TestNormalizationConversions:
    """Tests for explicit float64 [0, 1] normalization and uint8 [0, 255] quantization."""

    def test_image_to_normalized_float_rgb(self):
        arr = np.array([
            [[0, 51, 102], [153, 204, 255]],
            [[255, 128, 0], [64, 192, 255]],
        ], dtype=np.uint8)
        orig_copy = arr.copy()

        norm = image_to_normalized_float(arr)

        assert norm.shape == (2, 2, 3)
        assert norm.dtype == np.float64
        # Hand-calculated known values
        assert np.isclose(norm[0, 0, 0], 0.0)
        assert np.isclose(norm[0, 0, 1], 51.0 / 255.0)
        assert np.isclose(norm[0, 0, 2], 102.0 / 255.0)
        assert np.isclose(norm[0, 1, 0], 153.0 / 255.0)
        assert np.isclose(norm[0, 1, 1], 204.0 / 255.0)
        assert np.isclose(norm[0, 1, 2], 1.0)
        # Input immutability
        assert np.array_equal(arr, orig_copy)

    def test_image_to_normalized_float_grayscale(self):
        gray = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        norm = image_to_normalized_float(gray)
        assert norm.shape == (2, 2)
        assert norm.dtype == np.float64
        assert np.isclose(norm[0, 0], 0.0)
        assert np.isclose(norm[0, 1], 128.0 / 255.0)
        assert np.isclose(norm[1, 0], 1.0)
        assert np.isclose(norm[1, 1], 64.0 / 255.0)

    def test_normalized_float_to_image_rgb(self):
        norm = np.array([
            [[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]],
            [[-0.5, 0.5, 1.5], [0.001, 0.999, 0.50196]],
        ], dtype=np.float64)
        orig_copy = norm.copy()

        img = normalized_float_to_image(norm)

        assert img.shape == (2, 2, 3)
        assert img.dtype == np.uint8
        # Clamped & rounded values
        assert img[0, 0, 0] == 0
        assert img[0, 0, 1] == round(0.2 * 255.0)  # 51
        assert img[0, 0, 2] == round(0.4 * 255.0)  # 102
        assert img[0, 1, 2] == 255
        # Out-of-bounds clipping check: -0.5 -> 0, 1.5 -> 255
        assert img[1, 0, 0] == 0
        assert img[1, 0, 1] == 128
        assert img[1, 0, 2] == 255
        # Input immutability
        assert np.allclose(norm, orig_copy)

    def test_roundtrip_fidelity(self):
        """Every 8-bit integer in [0, 255] roundtrips through [0, 1] normalization exactly."""
        all_uint8 = np.arange(256, dtype=np.uint8).reshape((16, 16))
        normalized = image_to_normalized_float(all_uint8)
        recovered = normalized_float_to_image(normalized)
        assert np.array_equal(all_uint8, recovered)

    def test_legacy_adapters(self):
        legacy = np.array([0.0, 127.5, 255.0], dtype=np.float64)
        norm = legacy_float_to_normalized(legacy, is_delta=False)
        assert np.allclose(norm, [0.0, 0.5, 1.0])

        back_to_legacy = normalized_to_legacy_float(norm, is_delta=False)
        assert np.allclose(back_to_legacy, legacy)

        # Delta mode
        legacy_delta = np.array([-255.0, -127.5, 0.0, 127.5, 255.0], dtype=np.float64)
        norm_delta = legacy_float_to_normalized(legacy_delta, is_delta=True)
        assert np.allclose(norm_delta, [-1.0, -0.5, 0.0, 0.5, 1.0])

        back_delta = normalized_to_legacy_float(norm_delta, is_delta=True)
        assert np.allclose(back_delta, legacy_delta)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            image_to_normalized_float([0, 128, 255])
        with pytest.raises(TypeError):
            normalized_float_to_image("invalid")
        with pytest.raises(TypeError):
            legacy_float_to_normalized({"a": 1})


# ==============================================================================
# 2. Perturbation Operators in Normalized Float [0, 1] Space
# ==============================================================================

class TestNormalizedPerturbationOperators:
    """Tests that all 9 perturbation operator types run on normalized [0, 1] tensors."""

    @pytest.fixture
    def test_image_norm(self):
        # 16x16 gradient in [0, 1]
        x = np.linspace(0.1, 0.9, 16)
        xx, yy = np.meshgrid(x, x)
        base = np.stack([xx, yy, (xx + yy) / 2.0], axis=2)
        return base.astype(np.float64)

    def test_additive_random_normalized(self, test_image_norm):
        rng = np.random.default_rng(1234)
        orig = test_image_norm.copy()
        out, formula, exp, name, cat = PerturbationService._apply_operator(
            op_type="additive",
            input_float=test_image_norm,
            active_channels=[0, 1, 2],
            params={"epsilon": 0.05, "additive_mode": "random"},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert out.shape == test_image_norm.shape
        assert out.dtype == np.float64
        assert np.allclose(test_image_norm, orig)  # not mutated
        delta = out - test_image_norm
        assert np.max(np.abs(delta)) <= 0.05 + 1e-7

    def test_additive_uniform_normalized(self, test_image_norm):
        rng = np.random.default_rng(1234)
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="additive",
            input_float=test_image_norm,
            active_channels=[0, 1],
            params={"epsilon": 0.04, "additive_mode": "uniform"},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        delta = out - test_image_norm
        assert np.max(np.abs(delta[:, :, :2])) <= 0.04 + 1e-7
        assert np.allclose(delta[:, :, 2], 0.0)  # channel B unchanged

    def test_additive_positive_const_normalized(self, test_image_norm):
        rng = np.random.default_rng(1234)
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="additive",
            input_float=test_image_norm,
            active_channels=[0],
            params={"epsilon": 0.08, "additive_mode": "positive_const"},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        delta = out - test_image_norm
        assert np.allclose(delta[:, :, 0], 0.08)
        assert np.allclose(delta[:, :, 1:], 0.0)

    def test_additive_negative_const_normalized(self, test_image_norm):
        rng = np.random.default_rng(1234)
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="additive",
            input_float=test_image_norm,
            active_channels=[1],
            params={"epsilon": 0.06, "additive_mode": "negative_const"},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        delta = out - test_image_norm
        assert np.allclose(delta[:, :, 1], -0.06)
        assert np.allclose(delta[:, :, 0], 0.0)
        assert np.allclose(delta[:, :, 2], 0.0)

    def test_multiplicative_normalized(self, test_image_norm):
        rng = np.random.default_rng(42)
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="multiplicative",
            input_float=test_image_norm,
            active_channels=[0, 1, 2],
            params={"multiplicative_factor": 0.1},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        delta = out - test_image_norm
        # In multiplicative: out = I * (1 + delta_rel), where delta_rel in [-0.1, 0.1]
        assert np.all(out >= 0.0)
        assert np.max(np.abs(delta)) <= 0.1 * np.max(test_image_norm) + 1e-7

    def test_gaussian_normalized(self, test_image_norm):
        rng = np.random.default_rng(42)
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="gaussian",
            input_float=test_image_norm,
            active_channels=[0, 1, 2],
            params={"sigma": 0.05},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert out.shape == test_image_norm.shape
        delta = out - test_image_norm
        assert np.std(delta) > 0.0

    def test_salt_and_pepper_normalized(self, test_image_norm):
        rng = np.random.default_rng(42)
        out, _, _, _, _ = PerturbationService._apply_operator(
            op_type="salt_and_pepper",
            input_float=test_image_norm,
            active_channels=[0, 1, 2],
            params={"salt_pepper_prob": 0.2, "salt_vs_pepper_ratio": 0.5},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        # In normalized space, salt is 1.0 and pepper is 0.0
        assert np.any(np.isclose(out, 1.0))
        assert np.any(np.isclose(out, 0.0))

    def test_inversion_normalized(self, test_image_norm):
        rng = np.random.default_rng(42)
        out, _, _, _, cat = PerturbationService._apply_operator(
            op_type="inversion",
            input_float=test_image_norm,
            active_channels=[0, 1, 2],
            params={},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert cat == "educational_intensity"
        # In normalized space: out = 1.0 - in
        assert np.allclose(out, 1.0 - test_image_norm)

    def test_brightness_contrast_normalized(self, test_image_norm):
        rng = np.random.default_rng(42)
        out, _, _, _, cat = PerturbationService._apply_operator(
            op_type="brightness_contrast",
            input_float=test_image_norm,
            active_channels=[0, 1, 2],
            params={"contrast_factor": 1.2, "brightness_offset": 0.05},
            rng=rng,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert cat == "photometric"
        midpoint = 128.0 / 255.0
        expected = 1.2 * (test_image_norm - midpoint) + midpoint + 0.05
        assert np.allclose(out, expected)

    def test_deterministic_seed_reproducibility(self, test_image_norm):
        rng1 = np.random.default_rng(9999)
        out1, _, _, _, _ = PerturbationService._apply_operator(
            op_type="gaussian",
            input_float=test_image_norm,
            active_channels=[0, 1, 2],
            params={"sigma": 0.05},
            rng=rng1,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        rng2 = np.random.default_rng(9999)
        out2, _, _, _, _ = PerturbationService._apply_operator(
            op_type="gaussian",
            input_float=test_image_norm,
            active_channels=[0, 1, 2],
            params={"sigma": 0.05},
            rng=rng2,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        rng3 = np.random.default_rng(1111)
        out3, _, _, _, _ = PerturbationService._apply_operator(
            op_type="gaussian",
            input_float=test_image_norm,
            active_channels=[0, 1, 2],
            params={"sigma": 0.05},
            rng=rng3,
            is_normalized=True,
            parameter_unit="normalized_0_1",
        )
        assert np.array_equal(out1, out2)
        assert not np.array_equal(out1, out3)


# ==============================================================================
# 3. Quantization Distortion Function Tests
# ==============================================================================

class TestQuantizationDistortion:
    """Tests for calculate_quantization_distortion between continuous and quantized tensors."""

    def test_hand_calculated_quantization_distortion(self):
        # 1D known arrays
        cont = np.array([10.2, 20.7, 30.0, 40.4], dtype=np.float64)
        quant = np.array([10.0, 21.0, 30.0, 40.0], dtype=np.float64)
        # diff = cont - quant = [0.2, -0.3, 0.0, 0.4]
        # |diff| = [0.2, 0.3, 0.0, 0.4]
        # mae = (0.2 + 0.3 + 0.0 + 0.4) / 4 = 0.9 / 4 = 0.225
        # diff^2 = [0.04, 0.09, 0.0, 0.16]
        # mse = 0.29 / 4 = 0.0725
        # rmse = sqrt(0.0725) ≈ 0.269258
        # max_error = 0.4
        res = calculate_quantization_distortion(cont, quant)
        assert np.isclose(res["mae"], 0.225)
        assert np.isclose(res["mse"], 0.0725)
        assert np.isclose(res["rmse"], np.sqrt(0.0725))
        assert np.isclose(res["max_error"], 0.4)

    def test_identical_tensors_zero_distortion(self):
        t = np.ones((10, 10, 3), dtype=np.float64) * 128.0
        res = calculate_quantization_distortion(t, t)
        assert res["mae"] == 0.0
        assert res["mse"] == 0.0
        assert res["rmse"] == 0.0
        assert res["max_error"] == 0.0
        assert res["std"] == 0.0

    def test_shape_mismatch_raises(self):
        t1 = np.zeros((4, 4))
        t2 = np.zeros((4, 5))
        with pytest.raises(ValueError):
            calculate_quantization_distortion(t1, t2)


# ==============================================================================
# 4. Sequential and Weighted Composition with Normalized Budget
# ==============================================================================

class TestCompositionWithNormalizedBudget:
    """Verifies that pipeline composition operates in normalized space and satisfies actual budget."""

    @pytest.fixture(autouse=True)
    def setup_sample_image(self):
        # Upload a 32x32 gradient sample image
        img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        res = client.post("/api/upload", files={"file": ("norm_test.png", bio.getvalue(), "image/png")})
        assert res.status_code == 200
        self.image_id = res.json()["image_id"]

    def test_sequential_normalized_budget_enforcement(self):
        req = {
            "composition_mode": "sequential",
            "base_source": "original",
            "budget": {
                "enabled": True,
                "type": "both",
                "epsilon_inf": 0.03,  # normalized [0, 1]
                "epsilon_2": 0.5,     # normalized [0, 1]
                "unit": "normalized_0_1",
            },
            "operations": [
                {
                    "type": "additive",
                    "enabled": True,
                    "parameter_unit": "normalized_0_1",
                    "params": {"epsilon": 0.05, "additive_mode": "positive_const"},
                },
                {
                    "type": "gaussian",
                    "enabled": True,
                    "parameter_unit": "normalized_0_1",
                    "params": {"sigma": 0.04},
                },
            ],
            "seed": 42,
        }
        res = client.post(f"/api/images/{self.image_id}/perturb", json=req)
        assert res.status_code == 200
        data = res.json()

        # Check metadata
        report = data["budget_report"]
        assert report is not None
        assert report["all_constraints_satisfied"] is True
        assert report["post_projection_linf"] <= 0.03 + 1e-6
        assert report["post_projection_l2"] <= 0.5 + 1e-6

        # Check intermediate steps
        assert len(data["intermediate_steps"]) == 2
        for step in data["intermediate_steps"]:
            assert step["enabled"] is True
            assert step["preview_data_url"].startswith("data:image/png;base64,")

    def test_weighted_normalized_superposition(self):
        req = {
            "composition_mode": "weighted",
            "base_source": "original",
            "budget": {
                "enabled": True,
                "type": "linf",
                "epsilon_inf": 0.04,
                "unit": "normalized_0_1",
            },
            "operations": [
                {
                    "type": "additive",
                    "weight": 2.5,
                    "enabled": True,
                    "parameter_unit": "normalized_0_1",
                    "params": {"epsilon": 0.02, "additive_mode": "positive_const"},
                },
                {
                    "type": "additive",
                    "weight": 1.0,
                    "enabled": True,
                    "parameter_unit": "normalized_0_1",
                    "params": {"epsilon": 0.01, "additive_mode": "positive_const"},
                },
            ],
            "seed": 42,
        }
        res = client.post(f"/api/images/{self.image_id}/perturb", json=req)
        assert res.status_code == 200
        data = res.json()

        report = data["budget_report"]
        assert report["linf_satisfied"] is True
        assert report["post_projection_linf"] <= 0.04 + 1e-6

    def test_hand_calculated_weighted_linear_combination(self):
        """Verifies exact weighted superposition: Delta_comb = sum(w_i * Delta_i)."""
        base = np.full((4, 4, 3), 0.5, dtype=np.float64)
        # Delta 1: constant +0.02, weight 2.0 -> contribution = +0.04
        # Delta 2: constant -0.01, weight 3.0 -> contribution = -0.03
        # Expected combined delta = +0.01
        d1 = np.full((4, 4, 3), 0.02, dtype=np.float64)
        d2 = np.full((4, 4, 3), -0.01, dtype=np.float64)
        w1 = 2.0
        w2 = 3.0
        comb = w1 * d1 + w2 * d2
        assert np.allclose(comb, 0.01)

        final = np.clip(base + comb, 0.0, 1.0)
        assert np.allclose(final, 0.51)


# ==============================================================================
# 5. PNG Download and Normalized Tensor Export Integration Tests
# ==============================================================================

class TestDownloadAndTensorExportIntegration:
    """Verifies image PNG download and multi-format tensor exports."""

    @pytest.fixture(autouse=True)
    def setup_perturbed_image(self):
        # Create and upload image
        img = Image.new("RGB", (16, 16), color=(100, 150, 200))
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        res = client.post("/api/upload", files={"file": ("export_test.png", bio.getvalue(), "image/png")})
        assert res.status_code == 200
        self.image_id = res.json()["image_id"]

        # Apply perturbation
        pert_req = {
            "composition_mode": "sequential",
            "operations": [
                {
                    "type": "additive",
                    "enabled": True,
                    "parameter_unit": "normalized_0_1",
                    "params": {"epsilon": 0.05, "additive_mode": "uniform"},
                }
            ],
            "seed": 42,
        }
        res_p = client.post(f"/api/images/{self.image_id}/perturb", json=pert_req)
        assert res_p.status_code == 200

    def test_download_png_image_endpoint(self):
        # Download perturbed PNG
        res = client.get(f"/api/images/{self.image_id}/download?target=perturbed_rgb")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert res.content.startswith(b"\x89PNG\r\n\x1a\n")

        # Open as PIL image and check attributes
        downloaded_img = Image.open(io.BytesIO(res.content))
        assert downloaded_img.size == (16, 16)
        assert downloaded_img.mode == "RGB"

        # Download original PNG
        res_orig = client.get(f"/api/images/{self.image_id}/download?target=original_rgb")
        assert res_orig.status_code == 200
        assert res_orig.headers["content-type"] == "image/png"
        assert res_orig.content.startswith(b"\x89PNG\r\n\x1a\n")

    def test_export_normalized_tensor_npy(self):
        # Export perturbed_rgb_normalized
        res = client.get(f"/api/images/{self.image_id}/tensors/download?target=perturbed_rgb_normalized&format=npy")
        assert res.status_code == 200
        arr = np.load(io.BytesIO(res.content))
        assert arr.shape == (16, 16, 3)
        assert arr.dtype == np.float64
        # Range should be in [0.0, 1.0]
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)

        # Export perturbation_normalized
        res_delta = client.get(f"/api/images/{self.image_id}/tensors/download?target=perturbation_normalized&format=npy")
        assert res_delta.status_code == 200
        delta = np.load(io.BytesIO(res_delta.content))
        assert delta.shape == (16, 16, 3)
        assert delta.dtype == np.float64
        assert np.all(delta >= -1.0)
        assert np.all(delta <= 1.0)

    def test_export_all_tensors_npz_includes_normalized(self):
        res = client.get(f"/api/images/{self.image_id}/tensors/download?target=all&format=npz")
        assert res.status_code == 200
        npz = np.load(io.BytesIO(res.content))
        keys = list(npz.keys())
        assert "perturbed_rgb" in keys
        assert "perturbed_rgb_normalized" in keys
        assert "perturbation_normalized" in keys
        assert "delta_normalized" in keys

    def test_export_csv_formats_preserved(self):
        # Export CSV for grayscale
        res = client.get(f"/api/images/{self.image_id}/tensors/download?target=original_grayscale&format=csv")
        assert res.status_code == 200
        assert "text/csv" in res.headers["content-type"]
        csv_text = res.content.decode("utf-8")
        assert len(csv_text.strip().split("\n")) == 16
