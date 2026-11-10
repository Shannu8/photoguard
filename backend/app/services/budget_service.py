"""
Phase 4: Mathematical Perturbation Budget Module.

Provides pure mathematical projection functions onto L_infinity and L2 balls:
1. L_infinity Budget Projection:
   ||Δ||_∞ <= ε_inf  ==>  Δ_projected = clip(Δ, -ε_inf, +ε_inf)
   Image projection:
   I_projected = clip(I + Δ_projected, 0, 1)  (or [0, 255])
   Δ_actual = I_projected - I

2. L2 Ball Projection:
   ||Δ||_2 <= ε_2  ==>  Δ_projected = Δ * min(1, ε_2 / (||Δ||_2 + 1e-12))
   Default evaluation: Euclidean/Frobenius norm across the entire flattened image tensor:
   ||Δ||_2 = sqrt(sum(Δ^2)).

3. Combined Dual Budget Projection:
   Projection order:
     Step 1: Linf projection (clamps each element |Δ_i| <= ε_inf).
     Step 2: L2 projection (scales entire tensor by c = min(1, ε_2 / ||Δ||_2) <= 1).
             Since c <= 1, |c * Δ_i| <= |Δ_i| <= ε_inf, preserving the Linf bound.
     Step 3: Verification that ||Δ_final||_∞ <= ε_inf and ||Δ_final||_2 <= ε_2.
     Step 4: If pixel clipping is applied, recomputes Δ_actual = I_projected - I
             and re-verifies that both constraints remain satisfied.

4. Budget Reporting:
   Measures pre- and post-projection norms, configured limits, satisfaction status,
   projection method, and clipping flags using actual measured values.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class BudgetService:
    """Pure mathematical budget projection service preserving tensor precision and shape."""

    @staticmethod
    def project_linf(delta: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Projects perturbation tensor onto L_infinity ball of radius epsilon:
        ||Δ'||_∞ <= ε  ==>  Δ' = clip(Δ, -ε, +ε)

        Preserves original float64 precision and multidimensional shape.
        Does not mutate the input tensor in place.
        """
        if not isinstance(delta, np.ndarray):
            raise TypeError(f"delta must be a numpy ndarray, got {type(delta).__name__}")
        if epsilon < 0.0:
            raise ValueError(f"L_infinity budget epsilon must be non-negative, got {epsilon}")
        if delta.size == 0:
            return delta.copy().astype(np.float64)

        return np.clip(delta.astype(np.float64, copy=True), -float(epsilon), float(epsilon))

    @staticmethod
    def project_l2(
        delta: np.ndarray,
        epsilon2: float,
        norm_scope: str = "entire_tensor",
    ) -> np.ndarray:
        """
        Projects perturbation tensor onto L2 ball of radius epsilon2:
        ||Δ'||_2 <= ε_2  ==>  Δ' = Δ * min(1, ε_2 / (||Δ||_2 + 1e-12))

        Default evaluation: Euclidean/Frobenius norm across the entire image tensor:
        ||Δ||_2 = sqrt(sum(Δ^2)).
        Supports 'entire_tensor' (default), 'per_channel', and 'per_pixel' scopes.
        Preserves original multidimensional shape and numeric stability.
        Safely handles zero-norm perturbations without division by zero.
        Does not mutate the input tensor in place.
        """
        if not isinstance(delta, np.ndarray):
            raise TypeError(f"delta must be a numpy ndarray, got {type(delta).__name__}")
        if epsilon2 < 0.0:
            raise ValueError(f"L2 budget epsilon2 must be non-negative, got {epsilon2}")
        if delta.size == 0:
            return delta.copy().astype(np.float64)

        d_float = delta.astype(np.float64, copy=True)

        if norm_scope == "entire_tensor":
            current_l2 = float(np.sqrt(np.sum(d_float ** 2)))
            if current_l2 <= epsilon2 or current_l2 < 1e-12:
                return d_float

            scale_factor = float(epsilon2) / (current_l2 + 1e-12)
            projected = d_float * scale_factor

            # Numerical stability guard: ensure projected norm <= epsilon2
            post_l2 = float(np.sqrt(np.sum(projected ** 2)))
            if post_l2 > epsilon2 + 1e-12:
                projected = projected * (float(epsilon2) / (post_l2 + 1e-12))
            return projected

        elif norm_scope == "per_channel":
            if d_float.ndim == 3 and d_float.shape[2] > 1:
                projected = d_float.copy()
                for c in range(d_float.shape[2]):
                    ch_data = d_float[:, :, c]
                    ch_l2 = float(np.sqrt(np.sum(ch_data ** 2)))
                    if ch_l2 > epsilon2 and ch_l2 > 1e-12:
                        projected[:, :, c] = ch_data * (float(epsilon2) / (ch_l2 + 1e-12))
                return projected
            else:
                return BudgetService.project_l2(d_float, epsilon2, norm_scope="entire_tensor")

        elif norm_scope == "per_pixel":
            if d_float.ndim == 3:
                pixel_l2 = np.sqrt(np.sum(d_float ** 2, axis=2, keepdims=True))
                scale = np.where(
                    pixel_l2 > epsilon2,
                    float(epsilon2) / (pixel_l2 + 1e-12),
                    1.0,
                )
                return d_float * scale
            else:
                return np.clip(d_float, -float(epsilon2), float(epsilon2))

        else:
            raise ValueError(
                f"Unsupported norm_scope: '{norm_scope}'. "
                "Supported scopes: 'entire_tensor', 'per_channel', 'per_pixel'"
            )

    @classmethod
    def project_combined(
        cls,
        delta: np.ndarray,
        epsilon_inf: float,
        epsilon_2: float,
        norm_scope: str = "entire_tensor",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Projects perturbation tensor onto intersection of L_infinity and L2 balls.

        Mathematical projection order:
        1. Linf projection: clip each element to [-epsilon_inf, +epsilon_inf].
        2. L2 projection: scale tensor by c = min(1, epsilon_2 / ||d||_2).
           Because c <= 1, |c * d_i| <= |d_i| <= epsilon_inf, preserving the Linf bound.
        3. Strict dual verification: both ||Δ||_∞ <= ε_inf and ||Δ||_2 <= ε_2.
        """
        if not isinstance(delta, np.ndarray):
            raise TypeError(f"delta must be a numpy ndarray, got {type(delta).__name__}")
        if epsilon_inf < 0.0:
            raise ValueError(f"L_infinity epsilon must be non-negative, got {epsilon_inf}")
        if epsilon_2 < 0.0:
            raise ValueError(f"L2 epsilon must be non-negative, got {epsilon_2}")

        d_orig = delta.astype(np.float64, copy=True)
        orig_linf = float(np.max(np.abs(d_orig))) if d_orig.size > 0 else 0.0
        orig_l2 = float(np.sqrt(np.sum(d_orig ** 2))) if d_orig.size > 0 else 0.0

        # Step 1: Linf projection
        d_linf = cls.project_linf(d_orig, epsilon_inf)

        # Step 2: L2 projection
        d_comb = cls.project_l2(d_linf, epsilon_2, norm_scope=norm_scope)

        # Step 3: Verification
        final_linf = float(np.max(np.abs(d_comb))) if d_comb.size > 0 else 0.0
        final_l2 = float(np.sqrt(np.sum(d_comb ** 2))) if d_comb.size > 0 else 0.0

        # Safety iteration if numerical rounding slightly exceeded Linf
        if final_linf > epsilon_inf + 1e-12:
            d_comb = cls.project_linf(d_comb, epsilon_inf)
            final_linf = float(np.max(np.abs(d_comb)))
            final_l2 = float(np.sqrt(np.sum(d_comb ** 2)))

        was_projected = (orig_linf > epsilon_inf + 1e-7) or (orig_l2 > epsilon_2 + 1e-7)

        report = {
            "projection_method": "both",
            "norm_scope": norm_scope,
            "configured_epsilon_inf": epsilon_inf,
            "configured_epsilon_2": epsilon_2,
            "epsilon_linf": epsilon_inf,
            "epsilon_l2": epsilon_2,
            "original_linf": orig_linf,
            "original_l2": orig_l2,
            "pre_projection_linf": orig_linf,
            "pre_projection_l2": orig_l2,
            "projected_linf": final_linf,
            "projected_l2": final_l2,
            "post_projection_linf": final_linf,
            "post_projection_l2": final_l2,
            "linf_satisfied": bool(final_linf <= epsilon_inf + 1e-7),
            "l2_satisfied": bool(final_l2 <= epsilon_2 + 1e-7),
            "all_constraints_satisfied": bool(
                final_linf <= epsilon_inf + 1e-7 and final_l2 <= epsilon_2 + 1e-7
            ),
            "projection_applied": was_projected,
            "projected": was_projected,
            "clipping_applied": False,
            "pixel_clipping_corrected": False,
        }

        return d_comb, report

    @classmethod
    def project_linf_image(
        cls,
        image: np.ndarray,
        delta: np.ndarray,
        epsilon_inf: float,
        clip_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Given original image I and perturbation delta:
          delta_projected = clip(delta, -epsilon_inf, +epsilon_inf)
          I_projected = clip(I + delta_projected, clip_range[0], clip_range[1])
          actual_delta = I_projected - I

        Validates inputs and shapes.
        Preserves image dimensions and channels.
        Avoids modifying input tensors in place.
        Returns (I_projected, actual_delta, budget_report).
        """
        if not isinstance(image, np.ndarray) or not isinstance(delta, np.ndarray):
            raise TypeError("Both image and delta must be numpy ndarrays")
        if image.shape != delta.shape:
            raise ValueError(f"Shape mismatch: image {image.shape} vs delta {delta.shape}")
        if epsilon_inf < 0.0:
            raise ValueError(f"L_infinity epsilon must be non-negative, got {epsilon_inf}")

        img_float = image.astype(np.float64, copy=True)
        d_float = delta.astype(np.float64, copy=True)

        orig_linf = float(np.max(np.abs(d_float))) if d_float.size > 0 else 0.0
        orig_l2 = float(np.sqrt(np.sum(d_float ** 2))) if d_float.size > 0 else 0.0

        delta_projected = cls.project_linf(d_float, epsilon_inf)
        img_projected = np.clip(img_float + delta_projected, clip_range[0], clip_range[1])
        actual_delta = img_projected - img_float

        final_linf = float(np.max(np.abs(actual_delta))) if actual_delta.size > 0 else 0.0
        final_l2 = float(np.sqrt(np.sum(actual_delta ** 2))) if actual_delta.size > 0 else 0.0

        clipping_applied = not np.allclose(delta_projected, actual_delta, atol=1e-12)
        was_projected = (orig_linf > epsilon_inf + 1e-7) or clipping_applied

        report = {
            "projection_method": "linf",
            "norm_scope": "entire_tensor",
            "configured_epsilon_inf": epsilon_inf,
            "configured_epsilon_2": None,
            "epsilon_linf": epsilon_inf,
            "epsilon_l2": None,
            "original_linf": orig_linf,
            "original_l2": orig_l2,
            "pre_projection_linf": orig_linf,
            "pre_projection_l2": orig_l2,
            "projected_linf": final_linf,
            "projected_l2": final_l2,
            "post_projection_linf": final_linf,
            "post_projection_l2": final_l2,
            "linf_satisfied": bool(final_linf <= epsilon_inf + 1e-7),
            "l2_satisfied": True,
            "all_constraints_satisfied": bool(final_linf <= epsilon_inf + 1e-7),
            "projection_applied": was_projected,
            "projected": was_projected,
            "clipping_applied": clipping_applied,
            "pixel_clipping_corrected": clipping_applied,
        }

        return img_projected, actual_delta, report

    @classmethod
    def apply_image_budget(
        cls,
        image: np.ndarray,
        delta: np.ndarray,
        budget_type: str = "none",
        epsilon_inf: Optional[float] = None,
        epsilon_2: Optional[float] = None,
        l0_threshold: float = 1e-4,
        max_changed_pixel_ratio: Optional[float] = None,
        norm_scope: str = "entire_tensor",
        clip_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Unified image-level perturbation budget enforcement.
        Applies Linf, L2, or combined dual projection to delta, clips image,
        recomputes actual delta relative to base image, and measures final compliance.
        """
        if not isinstance(image, np.ndarray) or not isinstance(delta, np.ndarray):
            raise TypeError("Both image and delta must be numpy ndarrays")
        if image.shape != delta.shape:
            raise ValueError(f"Shape mismatch: image {image.shape} vs delta {delta.shape}")

        b_type = (budget_type or "none").lower()
        if b_type not in ["none", "linf", "l2", "both"]:
            raise ValueError(f"Unsupported budget type: '{budget_type}'. Expected: 'none', 'linf', 'l2', 'both'")

        img_float = image.astype(np.float64, copy=True)
        d_float = delta.astype(np.float64, copy=True)

        orig_linf = float(np.max(np.abs(d_float))) if d_float.size > 0 else 0.0
        orig_l2 = float(np.sqrt(np.sum(d_float ** 2))) if d_float.size > 0 else 0.0

        was_projected = False

        if b_type == "linf":
            if epsilon_inf is None or epsilon_inf < 0.0:
                raise ValueError(f"epsilon_inf must be non-negative for linf budget, got {epsilon_inf}")
            proj_delta = cls.project_linf(d_float, epsilon_inf)
            if orig_linf > epsilon_inf + 1e-7:
                was_projected = True

        elif b_type == "l2":
            if epsilon_2 is None or epsilon_2 < 0.0:
                raise ValueError(f"epsilon_2 must be non-negative for l2 budget, got {epsilon_2}")
            proj_delta = cls.project_l2(d_float, epsilon_2, norm_scope=norm_scope)
            if orig_l2 > epsilon_2 + 1e-7:
                was_projected = True

        elif b_type == "both":
            if epsilon_inf is None or epsilon_inf < 0.0:
                raise ValueError(f"epsilon_inf must be non-negative for dual budget, got {epsilon_inf}")
            if epsilon_2 is None or epsilon_2 < 0.0:
                raise ValueError(f"epsilon_2 must be non-negative for dual budget, got {epsilon_2}")
            proj_delta, _ = cls.project_combined(
                d_float, epsilon_inf=epsilon_inf, epsilon_2=epsilon_2, norm_scope=norm_scope
            )
            if (orig_linf > epsilon_inf + 1e-7) or (orig_l2 > epsilon_2 + 1e-7):
                was_projected = True

        else:  # "none"
            proj_delta = d_float.copy()

        # Apply pixel clipping
        img_projected = np.clip(img_float + proj_delta, clip_range[0], clip_range[1])
        actual_delta = img_projected - img_float

        clipping_applied = not np.allclose(proj_delta, actual_delta, atol=1e-12)

        # Actual final norms measured directly on actual_delta
        final_linf = float(np.max(np.abs(actual_delta))) if actual_delta.size > 0 else 0.0
        final_l2 = float(np.sqrt(np.sum(actual_delta ** 2))) if actual_delta.size > 0 else 0.0

        linf_satisfied = (
            bool(final_linf <= epsilon_inf + 1e-7)
            if (b_type in ["linf", "both"] and epsilon_inf is not None)
            else True
        )
        l2_satisfied = (
            bool(final_l2 <= epsilon_2 + 1e-7)
            if (b_type in ["l2", "both"] and epsilon_2 is not None)
            else True
        )

        # L0 changed pixel ratio calculation
        if actual_delta.ndim == 3 and actual_delta.shape[2] >= 3:
            pixel_max_abs = np.max(np.abs(actual_delta[:, :, :3]), axis=2)
            total_pixels = int(actual_delta.shape[0] * actual_delta.shape[1])
            changed_pixels = int(np.sum(pixel_max_abs > l0_threshold))
        else:
            total_pixels = int(actual_delta.size)
            changed_pixels = int(np.sum(np.abs(actual_delta) > l0_threshold))

        changed_ratio = float(changed_pixels / total_pixels) if total_pixels > 0 else 0.0
        ratio_satisfied = (
            bool(changed_ratio <= max_changed_pixel_ratio + 1e-7)
            if max_changed_pixel_ratio is not None
            else True
        )

        report = {
            "budget_type": b_type,
            "projection_method": b_type,
            "enabled": (b_type != "none"),
            "norm_scope": norm_scope,
            "configured_epsilon_inf": epsilon_inf,
            "configured_epsilon_2": epsilon_2,
            "epsilon_linf": epsilon_inf,
            "epsilon_l2": epsilon_2,
            "original_linf": orig_linf,
            "original_l2": orig_l2,
            "pre_projection_linf": orig_linf,
            "pre_projection_l2": orig_l2,
            "projected_linf": final_linf,
            "projected_l2": final_l2,
            "post_projection_linf": final_linf,
            "post_projection_l2": final_l2,
            "linf_satisfied": linf_satisfied,
            "l2_satisfied": l2_satisfied,
            "all_constraints_satisfied": bool(linf_satisfied and l2_satisfied and ratio_satisfied),
            "projection_applied": was_projected,
            "projected": was_projected,
            "clipping_applied": clipping_applied,
            "pixel_clipping_corrected": clipping_applied,
            "changed_pixels_count": changed_pixels,
            "total_pixels_count": total_pixels,
            "changed_pixel_ratio": changed_ratio,
            "max_changed_pixel_ratio": max_changed_pixel_ratio,
            "ratio_satisfied": ratio_satisfied,
        }

        return img_projected, actual_delta, report

    @classmethod
    def apply_budget(
        cls,
        delta: np.ndarray,
        budget_type: str,
        epsilon_linf: Optional[float] = None,
        epsilon_l2: Optional[float] = None,
        norm_scope: str = "entire_tensor",
        base_image: Optional[np.ndarray] = None,
        clip_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies configured budget projections (linf, l2, both, or none).
        Preserves backward compatibility with existing services and tests while
        providing rich, measured compliance reporting.
        """
        if base_image is not None and clip_range is not None:
            _, actual_delta, report = cls.apply_image_budget(
                image=base_image,
                delta=delta,
                budget_type=budget_type,
                epsilon_inf=epsilon_linf,
                epsilon_2=epsilon_l2,
                norm_scope=norm_scope,
                clip_range=clip_range,
            )
            return actual_delta, report

        d_orig = delta.astype(np.float64, copy=True)
        orig_linf = float(np.max(np.abs(d_orig))) if d_orig.size > 0 else 0.0
        orig_l2 = float(np.sqrt(np.sum(d_orig ** 2))) if d_orig.size > 0 else 0.0

        projected = d_orig.copy()
        was_projected = False

        b_type = (budget_type or "none").lower()

        if b_type in ["linf", "both"] and epsilon_linf is not None and epsilon_linf >= 0:
            projected = cls.project_linf(projected, epsilon_linf)
            if orig_linf > epsilon_linf + 1e-7:
                was_projected = True

        if b_type in ["l2", "both"] and epsilon_l2 is not None and epsilon_l2 >= 0:
            post_linf_l2 = float(np.sqrt(np.sum(projected ** 2)))
            projected = cls.project_l2(projected, epsilon_l2, norm_scope=norm_scope)
            if post_linf_l2 > epsilon_l2 + 1e-7:
                was_projected = True

            # If both were applied, re-verify Linf
            if b_type == "both":
                post_l2_linf = float(np.max(np.abs(projected))) if projected.size > 0 else 0.0
                if post_l2_linf > epsilon_linf + 1e-12:
                    projected = cls.project_linf(projected, epsilon_linf)

        final_linf = float(np.max(np.abs(projected))) if projected.size > 0 else 0.0
        final_l2 = float(np.sqrt(np.sum(projected ** 2))) if projected.size > 0 else 0.0

        linf_satisfied = (
            bool(final_linf <= epsilon_linf + 1e-7)
            if (b_type in ["linf", "both"] and epsilon_linf is not None)
            else True
        )
        l2_satisfied = (
            bool(final_l2 <= epsilon_l2 + 1e-7)
            if (b_type in ["l2", "both"] and epsilon_l2 is not None)
            else True
        )

        report = {
            "budget_type": b_type,
            "projection_method": b_type,
            "enabled": (b_type != "none"),
            "norm_scope": norm_scope,
            "configured_epsilon_inf": epsilon_linf,
            "configured_epsilon_2": epsilon_l2,
            "epsilon_linf": epsilon_linf,
            "epsilon_l2": epsilon_l2,
            "original_linf": orig_linf,
            "original_l2": orig_l2,
            "pre_projection_linf": orig_linf,
            "pre_projection_l2": orig_l2,
            "projected_linf": final_linf,
            "projected_l2": final_l2,
            "post_projection_linf": final_linf,
            "post_projection_l2": final_l2,
            "linf_satisfied": linf_satisfied,
            "l2_satisfied": l2_satisfied,
            "all_constraints_satisfied": bool(linf_satisfied and l2_satisfied),
            "projected": was_projected,
            "projection_applied": was_projected,
            "clipping_applied": False,
            "pixel_clipping_corrected": False,
        }

        return projected, report


# Global singleton instance
budget_service = BudgetService()
