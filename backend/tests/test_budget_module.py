"""
Comprehensive Unit Tests for Phase 4 Step 2:
Mathematical Perturbation Budget Module and Schemas.

Covers:
1. Perturbation already within budget
2. L-infinity projection
3. L2 projection
4. Zero perturbation
5. Combined budgets (Linf + L2 dual projection)
6. Invalid epsilon values (raise ValueError)
7. Invalid shapes (raise ValueError)
8. Normalized input range [0, 1]
9. RGB (3D) and grayscale (2D) tensors
10. Numerical stability (zero norm, tiny norm, division by zero)
11. Input tensors remaining unchanged (immutability)
12. Final budget compliance after pixel clipping
13. Schema validation & serialization
"""

import math
import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.services.budget_service import BudgetService, budget_service
from backend.app.models.schemas import (
    BudgetConfig,
    PerturbationStep,
    PerturbationConfig,
    PerturbationRequest,
)


# ==============================================================================
# 1. Perturbation Already Within Budget
# ==============================================================================

class TestBudgetWithinBudget:
    def test_linf_within_budget_unchanged(self):
        delta = np.array([[0.01, -0.02], [0.03, -0.01]], dtype=np.float64)
        epsilon_inf = 0.05
        projected = BudgetService.project_linf(delta, epsilon_inf)
        np.testing.assert_allclose(projected, delta)

    def test_l2_within_budget_unchanged(self):
        # norm = sqrt(0.01^2 + 0.02^2) = sqrt(0.0005) ≈ 0.02236
        delta = np.array([0.01, 0.02], dtype=np.float64)
        epsilon_2 = 0.05
        projected = BudgetService.project_l2(delta, epsilon_2)
        np.testing.assert_allclose(projected, delta)

    def test_combined_within_budget_unchanged(self):
        delta = np.array([[0.02, -0.01], [0.01, 0.02]], dtype=np.float64)
        # Linf = 0.02 <= 0.04; L2 = sqrt(4*0.0004 + 0.0001*2) = sqrt(0.0010) ≈ 0.0316 <= 0.05
        projected, report = BudgetService.project_combined(delta, epsilon_inf=0.04, epsilon_2=0.05)
        np.testing.assert_allclose(projected, delta)
        assert report["all_constraints_satisfied"] is True
        assert report["projection_applied"] is False


# ==============================================================================
# 2. L-infinity Projection
# ==============================================================================

class TestLInfinityProjection:
    def test_hand_calculated_2x2_clamping(self):
        """
        Known 2x2 matrix:
        delta = [[0.08, -0.12],
                 [0.02, -0.01]]
        epsilon_inf = 0.05
        Expected:
        [[0.05, -0.05],
         [0.02, -0.01]]
        """
        delta = np.array([[0.08, -0.12], [0.02, -0.01]], dtype=np.float64)
        expected = np.array([[0.05, -0.05], [0.02, -0.01]], dtype=np.float64)
        projected = BudgetService.project_linf(delta, epsilon=0.05)
        np.testing.assert_allclose(projected, expected, atol=1e-12)
        assert np.max(np.abs(projected)) <= 0.05 + 1e-12

    def test_rgb_3d_tensor(self):
        delta = np.full((4, 4, 3), 0.10, dtype=np.float64)
        delta[0, 0, 0] = -0.15
        projected = BudgetService.project_linf(delta, epsilon=0.03137)  # 8/255
        assert np.max(np.abs(projected)) <= 0.03137 + 1e-12
        assert projected.shape == (4, 4, 3)

    def test_project_linf_image_without_clipping(self):
        img = np.full((2, 2), 0.5, dtype=np.float64)
        delta = np.array([[0.10, -0.20], [0.05, -0.05]], dtype=np.float64)
        eps = 0.05
        img_proj, actual_delta, report = BudgetService.project_linf_image(
            img, delta, epsilon_inf=eps, clip_range=(0.0, 1.0)
        )
        assert report["linf_satisfied"] is True
        assert report["projected_linf"] <= eps + 1e-12
        assert actual_delta.shape == (2, 2)


# ==============================================================================
# 3. L2 Projection
# ==============================================================================

class TestL2Projection:
    def test_hand_calculated_scaling(self):
        """
        delta = [0.03, 0.04] -> L2 norm = sqrt(0.0009 + 0.0016) = 0.05
        epsilon_2 = 0.025 (half)
        Expected scaled = [0.015, 0.020] -> new norm = 0.025
        """
        delta = np.array([0.03, 0.04], dtype=np.float64)
        projected = BudgetService.project_l2(delta, epsilon2=0.025)
        expected = np.array([0.015, 0.020], dtype=np.float64)
        np.testing.assert_allclose(projected, expected, atol=1e-12)
        assert np.sqrt(np.sum(projected ** 2)) <= 0.025 + 1e-12

    def test_l2_across_entire_tensor_default(self):
        """L2 projection default norm scope is the entire tensor (H x W x C)."""
        rng = np.random.default_rng(123)
        delta = rng.uniform(-0.1, 0.1, size=(6, 6, 3))
        current_l2 = np.sqrt(np.sum(delta ** 2))
        target_l2 = current_l2 * 0.4
        projected = BudgetService.project_l2(delta, epsilon2=target_l2, norm_scope="entire_tensor")
        assert np.sqrt(np.sum(projected ** 2)) == pytest.approx(target_l2, rel=1e-6)

    def test_l2_per_channel_scope(self):
        delta = np.zeros((2, 2, 3), dtype=np.float64)
        # R channel norm = sqrt(4 * 1.0) = 2.0; G channel norm = sqrt(4 * 0.25) = 1.0
        delta[:, :, 0] = 1.0
        delta[:, :, 1] = 0.5
        projected = BudgetService.project_l2(delta, epsilon2=1.0, norm_scope="per_channel")
        # R should be scaled by 0.5; G should be unscaled (norm 1.0 <= 1.0)
        np.testing.assert_allclose(projected[:, :, 0], 0.5, atol=1e-12)
        np.testing.assert_allclose(projected[:, :, 1], 0.5, atol=1e-12)


# ==============================================================================
# 4. Zero Perturbation & Numerical Stability
# ==============================================================================

class TestZeroAndNumericalStability:
    def test_zero_delta_linf(self):
        delta = np.zeros((4, 4, 3), dtype=np.float64)
        projected = BudgetService.project_linf(delta, epsilon=0.05)
        np.testing.assert_allclose(projected, 0.0)

    def test_zero_delta_l2_safe_from_division_by_zero(self):
        delta = np.zeros((4, 4, 3), dtype=np.float64)
        projected = BudgetService.project_l2(delta, epsilon2=0.05)
        np.testing.assert_allclose(projected, 0.0)
        assert not np.isnan(projected).any()
        assert not np.isinf(projected).any()

    def test_tiny_delta_underflow_safety(self):
        delta = np.full((2, 2), 1e-15, dtype=np.float64)
        projected = BudgetService.project_l2(delta, epsilon2=0.01)
        assert not np.isnan(projected).any()
        assert np.sqrt(np.sum(projected ** 2)) <= 0.01

    def test_huge_delta_scaling(self):
        delta = np.full((2, 2), 1e6, dtype=np.float64)
        projected = BudgetService.project_l2(delta, epsilon2=1.0)
        assert np.sqrt(np.sum(projected ** 2)) <= 1.0 + 1e-12
        assert not np.isinf(projected).any()


# ==============================================================================
# 5. Combined Dual Budgets (Linf + L2)
# ==============================================================================

class TestCombinedBudgets:
    def test_both_constraints_strictly_satisfied(self):
        """
        delta = [0.10, 0.05]
        epsilon_inf = 0.06, epsilon_2 = 0.05
        Linf projection clamps to [0.06, 0.05] (norm ≈ 0.0781).
        L2 projection scales by (0.05 / 0.0781) ≈ 0.6402.
        Result ≈ [0.03841, 0.03201].
        Linf ≈ 0.03841 <= 0.06; L2 ≈ 0.05 <= 0.05.
        """
        delta = np.array([0.10, 0.05], dtype=np.float64)
        eps_inf = 0.06
        eps_2 = 0.05
        projected, report = BudgetService.project_combined(delta, epsilon_inf=eps_inf, epsilon_2=eps_2)

        actual_linf = np.max(np.abs(projected))
        actual_l2 = np.sqrt(np.sum(projected ** 2))

        assert actual_linf <= eps_inf + 1e-12
        assert actual_l2 <= eps_2 + 1e-12
        assert report["linf_satisfied"] is True
        assert report["l2_satisfied"] is True
        assert report["all_constraints_satisfied"] is True
        assert report["projection_applied"] is True

    def test_projection_order_linf_then_l2_preserves_linf(self):
        """
        Scaling by c <= 1 can NEVER increase any element's absolute value.
        Therefore, Linf projection followed by L2 projection preserves Linf bound.
        """
        rng = np.random.default_rng(42)
        delta = rng.uniform(-1.0, 1.0, size=(10, 10, 3))
        eps_inf = 0.03137  # 8/255
        eps_2 = 0.20

        projected, report = BudgetService.project_combined(delta, eps_inf, eps_2)
        assert report["post_projection_linf"] <= eps_inf + 1e-7
        assert report["post_projection_l2"] <= eps_2 + 1e-7


# ==============================================================================
# 6. Invalid Inputs & Shape Validation
# ==============================================================================

class TestValidationAndErrors:
    def test_negative_epsilon_linf_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            BudgetService.project_linf(np.ones((2, 2)), epsilon=-0.01)

    def test_negative_epsilon_l2_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            BudgetService.project_l2(np.ones((2, 2)), epsilon2=-1.0)

    def test_non_ndarray_raises_type_error(self):
        with pytest.raises(TypeError):
            BudgetService.project_linf([[1, 2], [3, 4]], epsilon=0.5)

    def test_mismatched_image_and_delta_shapes_raises(self):
        img = np.zeros((4, 4, 3))
        delta = np.zeros((4, 5, 3))
        with pytest.raises(ValueError, match="Shape mismatch"):
            BudgetService.project_linf_image(img, delta, epsilon_inf=0.05)

    def test_invalid_norm_scope_raises(self):
        with pytest.raises(ValueError, match="Unsupported norm_scope"):
            BudgetService.project_l2(np.ones((2, 2)), epsilon2=1.0, norm_scope="invalid_scope")


# ==============================================================================
# 7. Immutability of Input Tensors
# ==============================================================================

class TestImmutability:
    def test_delta_not_modified_in_place_linf(self):
        delta = np.array([0.10, -0.15], dtype=np.float64)
        delta_copy = delta.copy()
        _ = BudgetService.project_linf(delta, epsilon=0.05)
        np.testing.assert_array_equal(delta, delta_copy)

    def test_delta_not_modified_in_place_l2(self):
        delta = np.array([3.0, 4.0], dtype=np.float64)
        delta_copy = delta.copy()
        _ = BudgetService.project_l2(delta, epsilon2=1.0)
        np.testing.assert_array_equal(delta, delta_copy)

    def test_image_not_modified_in_place(self):
        img = np.full((3, 3), 0.5, dtype=np.float64)
        delta = np.full((3, 3), 0.2, dtype=np.float64)
        img_copy = img.copy()
        delta_copy = delta.copy()
        _ = BudgetService.project_linf_image(img, delta, epsilon_inf=0.05)
        np.testing.assert_array_equal(img, img_copy)
        np.testing.assert_array_equal(delta, delta_copy)


# ==============================================================================
# 8. Pixel Clipping & Recomputed Budget Compliance
# ==============================================================================

class TestPixelClippingCorrection:
    def test_clipping_recomputes_actual_delta_and_satisfies_budget(self):
        """
        Original image has a bright pixel at 0.98.
        Perturbation is +0.08.
        With epsilon_inf = 0.05, projected delta is +0.05.
        Image becomes 0.98 + 0.05 = 1.03 -> clipped to 1.00.
        Actual delta = 1.00 - 0.98 = 0.02.
        Actual delta (0.02) satisfies Linf <= 0.05.
        clipping_applied flag is True.
        """
        img = np.array([[0.98, 0.02], [0.50, 0.50]], dtype=np.float64)
        delta = np.array([[0.08, -0.08], [0.03, -0.03]], dtype=np.float64)
        eps = 0.05

        img_proj, actual_delta, report = BudgetService.apply_image_budget(
            image=img,
            delta=delta,
            budget_type="linf",
            epsilon_inf=eps,
            clip_range=(0.0, 1.0),
        )

        assert report["clipping_applied"] is True
        assert report["linf_satisfied"] is True
        assert report["post_projection_linf"] <= eps + 1e-12
        # Pixel (0,0) was clipped to 1.0, so actual delta is 1.0 - 0.98 = 0.02
        assert actual_delta[0, 0] == pytest.approx(0.02, abs=1e-12)
        # Pixel (0,1) was clipped to 0.0, so actual delta is 0.0 - 0.02 = -0.02
        assert actual_delta[0, 1] == pytest.approx(-0.02, abs=1e-12)

    def test_max_changed_pixel_ratio_monitoring(self):
        img = np.full((4, 4), 0.5, dtype=np.float64)
        delta = np.zeros((4, 4), dtype=np.float64)
        # Change only 2 pixels out of 16 (ratio = 2/16 = 0.125)
        delta[0, 0] = 0.05
        delta[1, 1] = 0.05

        _, _, report_pass = BudgetService.apply_image_budget(
            image=img,
            delta=delta,
            budget_type="none",
            max_changed_pixel_ratio=0.20,
            l0_threshold=1e-4,
        )
        assert report_pass["ratio_satisfied"] is True
        assert report_pass["changed_pixel_ratio"] == pytest.approx(0.125)

        _, _, report_fail = BudgetService.apply_image_budget(
            image=img,
            delta=delta,
            budget_type="none",
            max_changed_pixel_ratio=0.10,
            l0_threshold=1e-4,
        )
        assert report_fail["ratio_satisfied"] is False


# ==============================================================================
# 9. Schema Validation & Serialization Tests
# ==============================================================================

class TestSchemasAndValidation:
    def test_budget_config_defaults(self):
        cfg = BudgetConfig(type="linf", unit="normalized_0_1")
        assert cfg.enabled is True
        assert cfg.epsilon_inf == pytest.approx(8.0 / 255.0)
        assert cfg.epsilon_linf == pytest.approx(8.0 / 255.0)
        assert cfg.norm_scope == "entire_tensor"

    def test_budget_config_syncs_aliases(self):
        # Passed epsilon_linf, should sync to epsilon_inf
        cfg1 = BudgetConfig(type="linf", epsilon_linf=0.04, unit="normalized_0_1")
        assert cfg1.epsilon_inf == 0.04

        # Passed epsilon_2, should sync to epsilon_l2
        cfg2 = BudgetConfig(type="l2", epsilon_2=0.5, unit="normalized_0_1")
        assert cfg2.epsilon_l2 == 0.5

    def test_budget_config_rejects_normalized_values_above_one(self):
        """Do not silently interpret values > 1 as 8-bit values under normalized units."""
        with pytest.raises(ValidationError, match="exceeds 1.0 for unit='normalized_0_1'"):
            BudgetConfig(type="linf", epsilon_inf=8.0, unit="normalized_0_1")

    def test_budget_config_allows_pixel_values_above_one_when_explicit(self):
        cfg = BudgetConfig(type="linf", epsilon_inf=8.0, unit="pixel_0_255")
        assert cfg.epsilon_inf == 8.0

    def test_perturbation_step_empty_channels_raises(self):
        with pytest.raises(ValidationError, match="channels list cannot be empty"):
            PerturbationStep(type="additive", channels=[])

    def test_perturbation_step_normalized_additive_over_one_raises(self):
        with pytest.raises(ValidationError, match="exceeds 1.0 for parameter_unit='normalized_0_1'"):
            PerturbationStep(
                type="additive",
                parameter_unit="normalized_0_1",
                params={"epsilon": 10.0},
            )

    def test_perturbation_step_invalid_salt_pepper_prob_raises(self):
        with pytest.raises(ValidationError, match="salt_pepper_prob must be between 0.0 and 1.0"):
            PerturbationStep(
                type="salt_and_pepper",
                params={"salt_pepper_prob": 1.5},
            )

    def test_perturbation_config_roundtrip_serialization(self):
        cfg = PerturbationConfig(
            version="4.0",
            composition_mode="weighted",
            budget=BudgetConfig(type="linf", epsilon_inf=0.03137, unit="normalized_0_1"),
            operations=[
                PerturbationStep(
                    type="additive",
                    weight=2.0,
                    params={"epsilon": 5.0},
                    parameter_unit="pixel_0_255",
                )
            ],
            seed=999,
        )
        json_data = cfg.model_dump_json()
        loaded = PerturbationConfig.model_validate_json(json_data)
        assert loaded.version == "4.0"
        assert loaded.composition_mode == "weighted"
        assert loaded.seed == 999
        assert len(loaded.operations) == 1
        assert loaded.budget.epsilon_inf == pytest.approx(0.03137)

    def test_perturbation_request_unpacks_nested_config(self):
        raw_payload = {
            "config": {
                "composition_mode": "weighted",
                "seed": 777,
                "operations": [{"type": "additive", "params": {"epsilon": 2.0}}],
            }
        }
        req = PerturbationRequest.model_validate(raw_payload)
        assert req.composition_mode == "weighted"
        assert req.seed == 777
        assert len(req.operations) == 1

