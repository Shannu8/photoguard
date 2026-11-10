"""
Phase 4: Mathematical Image Perturbation Engine Service.

Supports:
- 6 pure mathematical perturbation operators:
  1. Bounded additive perturbation (||Δ||_∞ <= ε)
  2. Multiplicative scaling perturbation (I ⊙ (1 + α·ξ))
  3. Gaussian white noise (Δ ~ N(0, σ²))
  4. Salt-and-pepper impulse noise (p, r)
  5. Pixel inversion (educational negative: 255 - I)
  6. Brightness and contrast adjustment (c·(I - 128) + 128 + b)
- Two mathematically distinct composition modes:
  Mode 1: Sequential Composition (Markov chain: I_k = clip(P_k(I_{k-1}), 0, 255))
          with cumulative budget projection relative to base image at every step.
  Mode 2: Weighted Perturbation Combination (Δ_combined = sum(w_k * Δ_k) relative to base image)
          with global budget projection on combined delta tensor.
- Deterministic reproducibility with random seed (pipeline seed and per-step seeds).
- Formal Perturbation Budget Projection (L_infinity and Euclidean L2 ball projections across tensor).
- Full intermediate tensor preservation and multi-step localized pixel inspection.
- Educational context separation: distinguishing subtle noise from photographic transforms.
- L0 element vs spatial pixel change detection with configurable threshold.
- Quantization distortion analysis E_quant = Δ_uint8 - Δ_continuous.
"""

import uuid
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List
from fastapi import HTTPException

from app.services.image_service import StoredImage
from app.services.budget_service import BudgetService
from app.utils.image_utils import array_to_base64_png, generate_error_heatmap_base64
from app.utils.math_utils import (
    rgb_to_grayscale,
    calculate_mse,
    calculate_psnr,
    calculate_mae,
    calculate_norms_and_diff_stats,
    calculate_ssim,
    calculate_lpips_hook,
    normalize_to_01,
    denormalize_from_01,
    image_to_normalized_float,
    normalized_float_to_image,
    legacy_float_to_normalized,
    normalized_to_legacy_float,
    calculate_quantization_distortion,
)
from app.models.schemas import (
    PerturbationStep,
    PerturbationRequest,
    PerturbationResponse,
    PerturbationMetrics,
    ChannelMetrics,
    PerturbationStepResult,
    IntermediatePixelValue,
    PerturbationPixelInspection,
)


class PerturbationService:
    """Mathematical perturbation engine with sequential and weighted composition pipelines."""

    @staticmethod
    def _get_base_image(stored: StoredImage, base_source: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves base RGB and Grayscale arrays.
        Supports both uploaded original image and Phase 3 transformed image.
        """
        if base_source == "transformed":
            if stored.transformed_rgb_array is None:
                raise HTTPException(
                    status_code=400,
                    detail="No Phase 3 transformed image found. Please apply a transformation first or select 'original' as the base source."
                )
            base_rgb = stored.transformed_rgb_array.copy()
            base_gray = (
                stored.transformed_grayscale_array.copy()
                if stored.transformed_grayscale_array is not None
                else rgb_to_grayscale(base_rgb)
            )
        else:
            base_rgb = stored.rgb_array.copy()
            base_gray = stored.grayscale_array.copy()

        return base_rgb, base_gray

    @staticmethod
    def _resolve_budget_config(
        req: PerturbationRequest,
    ) -> Tuple[str, Optional[float], Optional[float], str]:
        """
        Resolves budget configuration with explicit units and legacy parameter adapter.
        Returns: (budget_type, epsilon_linf_pixel, epsilon_l2_pixel, effective_unit)
        """
        if req.budget is not None and req.budget.type != "none" and req.budget.enabled:
            b_type = req.budget.type
            unit = req.budget.unit or "pixel_0_255"
            eps_linf = req.budget.epsilon_inf if req.budget.epsilon_inf is not None else req.budget.epsilon_linf
            eps_l2 = req.budget.epsilon_2 if req.budget.epsilon_2 is not None else req.budget.epsilon_l2

            # Convert normalized [0, 1] to pixel [0, 255] for internal tensor projection
            if unit == "normalized_0_1":
                linf_pixel = (float(eps_linf) * 255.0) if eps_linf is not None else None
                l2_pixel = (float(eps_l2) * 255.0) if eps_l2 is not None else None
            else:
                linf_pixel = float(eps_linf) if eps_linf is not None else None
                l2_pixel = float(eps_l2) if eps_l2 is not None else None

            return b_type, linf_pixel, l2_pixel, unit

        # Legacy adapter: req.global_bound is on pixel [0, 255] scale
        if req.global_bound is not None and req.global_bound >= 0:
            return "linf", float(req.global_bound), None, "pixel_0_255"

        return "none", None, None, "pixel_0_255"

    @staticmethod
    def _apply_operator(
        op_type: str,
        input_float: np.ndarray,
        active_channels: List[int],
        params: Dict[str, Any],
        rng: np.random.Generator,
        is_normalized: bool = True,
        parameter_unit: str = "pixel_0_255",
    ) -> Tuple[np.ndarray, str, str, str, str]:
        """
        Applies a single mathematical perturbation operator on float64 input tensor.
        Executes natively in normalized float64 [0.0, 1.0] space when is_normalized=True.
        Returns: (perturbed_float, formula_latex, explanation, display_name, category)
        Does NOT clip to uint8 internally to preserve floating point precision.
        """
        h, w, c = input_float.shape
        perturbed_float = input_float.copy()
        delta_shape = (h, w, len(active_channels))

        formula_str = ""
        explanation_str = ""
        display_name = ""
        category = "bounded_noise"

        unit = params.get("parameter_unit", parameter_unit)

        if op_type == "additive":
            raw_eps = float(params.get("epsilon", 10.0))
            mode = str(params.get("additive_mode", "random"))
            display_name = "Bounded Additive Noise"
            category = "bounded_noise"

            if is_normalized:
                eps = raw_eps if unit == "normalized_0_1" else (raw_eps / 255.0)
            else:
                eps = raw_eps

            if mode == "random":
                raw_delta = rng.uniform(-eps, eps, size=delta_shape)
                formula_str = r"I' = \text{clip}(I + \Delta, 0, 1), \quad \Delta \sim \mathcal{U}(-\varepsilon, \varepsilon), \quad \|\Delta\|_\infty \le \varepsilon"
                explanation_str = f"Additive uniform random perturbation with bound ε = {eps:.4f} across selected channels."
            elif mode == "uniform" or mode == "positive_const":
                raw_delta = np.full(delta_shape, eps, dtype=np.float64)
                formula_str = r"I' = \text{clip}(I + \varepsilon, 0, 1), \quad \|\Delta\|_\infty = \varepsilon"
                explanation_str = f"Additive positive constant shift of +{eps:.4f} across selected channels."
            elif mode == "negative_const":
                raw_delta = np.full(delta_shape, -eps, dtype=np.float64)
                formula_str = r"I' = \text{clip}(I - \varepsilon, 0, 1)"
                explanation_str = f"Additive negative shift of -{eps:.4f} across selected channels."
            else:
                raw_delta = rng.uniform(-eps, eps, size=delta_shape)
                formula_str = r"I' = \text{clip}(I + \Delta, 0, 1)"
                explanation_str = f"Additive perturbation with bound ε = {eps:.4f}."

            for idx, ch_idx in enumerate(active_channels):
                perturbed_float[:, :, ch_idx] += raw_delta[:, :, idx]

        elif op_type == "multiplicative":
            alpha = float(params.get("multiplicative_factor", 0.1))
            display_name = "Multiplicative Perturbation"
            category = "bounded_noise"
            rel_delta = rng.uniform(-alpha, alpha, size=delta_shape)

            for idx, ch_idx in enumerate(active_channels):
                perturbed_float[:, :, ch_idx] = perturbed_float[:, :, ch_idx] * (1.0 + rel_delta[:, :, idx])

            formula_str = r"I' = \text{clip}(I \odot (1 + \Delta), 0, 1), \quad \Delta \sim \mathcal{U}(-\alpha, \alpha)"
            explanation_str = f"Multiplicative scaling perturbation with intensity factor α = {alpha:.2f} (±{alpha*100:.1f}% variation)."

        elif op_type == "gaussian":
            raw_sigma = float(params.get("sigma", 15.0))
            display_name = "Gaussian White Noise"
            category = "bounded_noise"

            if is_normalized:
                sigma = raw_sigma if unit == "normalized_0_1" else (raw_sigma / 255.0)
            else:
                sigma = raw_sigma

            gauss_noise = rng.normal(loc=0.0, scale=sigma, size=delta_shape)

            for idx, ch_idx in enumerate(active_channels):
                perturbed_float[:, :, ch_idx] += gauss_noise[:, :, idx]

            formula_str = r"I' = \text{clip}(I + \mathcal{N}(0, \sigma^2), 0, 1)"
            explanation_str = f"Additive Gaussian white noise with zero mean and standard deviation σ = {sigma:.4f}."

        elif op_type == "salt_and_pepper":
            p = float(params.get("salt_pepper_prob", 0.05))
            r = float(params.get("salt_vs_pepper_ratio", 0.5))
            display_name = "Salt-and-Pepper Impulse"
            category = "bounded_noise"

            salt_val = 1.0 if is_normalized else 255.0
            pepper_val = 0.0

            for ch_idx in active_channels:
                rand_mask = rng.random(size=(h, w))
                salt_mask = rand_mask < (p * r)
                pepper_mask = (rand_mask >= (p * r)) & (rand_mask < p)

                perturbed_float[salt_mask, ch_idx] = salt_val
                perturbed_float[pepper_mask, ch_idx] = pepper_val

            formula_str = r"I'(x, y) = \begin{cases} 1.0 & \text{prob } p \cdot r \\ 0.0 & \text{prob } p \cdot (1 - r) \\ I(x, y) & \text{prob } 1 - p \end{cases}"
            explanation_str = f"Impulse Salt-and-Pepper noise with probability p = {p:.3f} and salt ratio r = {r:.2f}."

        elif op_type == "inversion":
            display_name = "Pixel Inversion (Negative)"
            category = "educational_intensity"

            inv_max = 1.0 if is_normalized else 255.0
            for ch_idx in active_channels:
                perturbed_float[:, :, ch_idx] = inv_max - perturbed_float[:, :, ch_idx]

            formula_str = r"I' = 1.0 - I" if is_normalized else r"I' = 255 - I"
            explanation_str = "Educational photographic negative transformation: complement inversion."

        elif op_type == "brightness_contrast":
            c_factor = float(params.get("contrast_factor", 1.2))
            raw_b = float(params.get("brightness_offset", 10.0))
            display_name = "Brightness & Contrast"
            category = "photometric"

            if is_normalized:
                b_offset = raw_b if unit == "normalized_0_1" else (raw_b / 255.0)
                midpoint = 128.0 / 255.0
            else:
                b_offset = raw_b
                midpoint = 128.0

            for ch_idx in active_channels:
                perturbed_float[:, :, ch_idx] = c_factor * (perturbed_float[:, :, ch_idx] - midpoint) + midpoint + b_offset

            formula_str = r"I' = \text{clip}(c \cdot (I - 0.5) + 0.5 + b, 0, 1)" if is_normalized else r"I' = \text{clip}(c \cdot (I - 128) + 128 + b, 0, 255)"
            explanation_str = f"Photometric adjustment: contrast factor c = {c_factor:.2f}, brightness bias b = {b_offset:+.4f}."

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported perturbation method: {op_type}")

        return perturbed_float, formula_str, explanation_str, display_name, category

    @classmethod
    def apply_perturbation(
        cls, stored: StoredImage, req: PerturbationRequest
    ) -> PerturbationResponse:
        """
        Executes a multi-perturbation pipeline or single perturbation operation.
        Supports both Sequential Composition and Weighted Perturbation Combination modes.
        Enforces cumulative budget relative to base image at every step.
        """
        base_rgb, base_gray = cls._get_base_image(stored, req.base_source)
        h, w, c = base_rgb.shape

        # Initialize deterministic RNG if seed provided
        rng = np.random.default_rng(req.seed) if req.seed is not None else np.random.default_rng()

        channel_map = {"R": 0, "G": 1, "B": 2}

        # Resolve formal budget configuration with explicit units & legacy fallback
        b_type, eps_linf_px, eps_l2_px, b_unit = cls._resolve_budget_config(req)

        # Normalize operations list: support multi-step pipeline or legacy single-step
        if not req.operations:
            method_str = req.method or "additive"
            single_params = {
                "epsilon": req.epsilon,
                "additive_mode": req.additive_mode,
                "multiplicative_factor": req.multiplicative_factor,
                "sigma": req.sigma,
                "salt_pepper_prob": req.salt_pepper_prob,
                "salt_vs_pepper_ratio": req.salt_vs_pepper_ratio,
                "contrast_factor": req.contrast_factor,
                "brightness_offset": req.brightness_offset,
            }
            operations = [
                PerturbationStep(
                    id="step-1",
                    type=method_str,
                    enabled=True,
                    weight=1.0,
                    channels=req.channels,
                    params=single_params,
                )
            ]
        else:
            operations = req.operations

        base_float = base_rgb.astype(np.float64)
        base_norm = image_to_normalized_float(base_rgb)

        # Normalized budget bounds in [0, 1]
        eps_linf_norm = (eps_linf_px / 255.0) if eps_linf_px is not None else None
        eps_l2_norm = (eps_l2_px / 255.0) if eps_l2_px is not None else None

        # Tracking variables
        step_results: List[PerturbationStepResult] = []
        intermediate_rgb_list: List[np.ndarray] = []
        individual_delta_list: List[np.ndarray] = []
        pipeline_steps_meta: List[Dict[str, Any]] = []
        last_budget_report: Optional[Dict[str, Any]] = None

        overall_formula = ""
        overall_explanation = ""

        # ==============================================================================
        # MODE 1: Sequential Composition (Markov chain: I_k = clip(P_k(I_{k-1}), 0, 1))
        # Cumulative budget enforced relative to base image after every step in [0, 1].
        # ==============================================================================
        if req.composition_mode == "sequential":
            current_norm = base_norm.copy()
            current_uint8 = base_rgb.copy()

            for idx, op in enumerate(operations):
                step_idx = idx + 1
                op_id = op.id or f"step-{step_idx}"

                active_channels = [channel_map[ch] for ch in op.channels if ch in channel_map]
                if not active_channels:
                    active_channels = [0, 1, 2]

                if not op.enabled:
                    # Disabled operation: intermediate state remains identical to previous
                    step_metrics = PerturbationMetrics(
                        mse=0.0,
                        psnr=100.0,
                        ssim=1.0,
                        mae=0.0,
                        l1_norm=0.0,
                        l2_norm=0.0,
                        linf_norm=0.0,
                        max_abs_perturbation=0.0,
                        min_difference=0.0,
                        max_difference=0.0,
                        mean_difference=0.0,
                        std_diff=0.0,
                        l0_threshold=float(req.l0_threshold),
                        changed_elements_count=0,
                        total_elements_count=int(base_rgb.size),
                        changed_elements_percent=0.0,
                        changed_pixels_count=0,
                        total_pixels_count=int(base_rgb.shape[0] * base_rgb.shape[1]),
                        changed_pixels_percent=0.0,
                        modified_pixels_count=0,
                        modified_pixels_percent=0.0,
                        lpips=None,
                        per_channel=None,
                        quantization_distortion=None,
                    )
                    step_results.append(
                        PerturbationStepResult(
                            step_index=step_idx,
                            step_id=op_id,
                            type=op.type,
                            name=f"{op.type.capitalize()} (Disabled)",
                            category="bounded_noise",
                            enabled=False,
                            weight=op.weight,
                            formula_latex="I' = I",
                            preview_data_url=array_to_base64_png(current_uint8, is_grayscale=False),
                            intermediate_metrics=step_metrics,
                        )
                    )
                    intermediate_rgb_list.append(current_uint8)
                    individual_delta_list.append(np.zeros_like(base_float))
                    continue

                # Use individual step seed if provided, else use pipeline RNG
                step_rng = np.random.default_rng(op.seed) if op.seed is not None else rng

                # Apply operator on current normalized floating-point image [0, 1]
                pert_norm, formula_latex, explanation, display_name, category = cls._apply_operator(
                    op_type=op.type,
                    input_float=current_norm,
                    active_channels=active_channels,
                    params=op.params,
                    rng=step_rng,
                    is_normalized=True,
                    parameter_unit=getattr(op, "parameter_unit", "pixel_0_255"),
                )

                # Sequential Global Budget Enforcement in normalized [0, 1] space:
                # Calculate cumulative delta relative to base image and project onto budget ball
                raw_cum_delta = pert_norm - base_norm
                if b_type != "none":
                    proj_cum_delta, last_budget_report = BudgetService.apply_budget(
                        raw_cum_delta,
                        budget_type=b_type,
                        epsilon_linf=eps_linf_norm,
                        epsilon_l2=eps_l2_norm,
                    )
                    current_norm = np.clip(base_norm + proj_cum_delta, 0.0, 1.0)
                else:
                    current_norm = np.clip(pert_norm, 0.0, 1.0)

                actual_cum_delta = current_norm - base_norm
                actual_linf_norm = float(np.max(np.abs(actual_cum_delta))) if actual_cum_delta.size > 0 else 0.0
                actual_l2_norm = float(np.sqrt(np.sum(actual_cum_delta ** 2))) if actual_cum_delta.size > 0 else 0.0

                if last_budget_report is not None:
                    if b_unit == "pixel_0_255":
                        last_budget_report["post_projection_linf"] = actual_linf_norm * 255.0
                        last_budget_report["post_projection_l2"] = actual_l2_norm * 255.0
                        last_budget_report["pre_projection_linf"] = float(np.max(np.abs(raw_cum_delta))) * 255.0
                        last_budget_report["pre_projection_l2"] = float(np.sqrt(np.sum(raw_cum_delta ** 2))) * 255.0
                        last_budget_report["epsilon_linf"] = eps_linf_px
                        last_budget_report["epsilon_l2"] = eps_l2_px
                    else:
                        last_budget_report["post_projection_linf"] = actual_linf_norm
                        last_budget_report["post_projection_l2"] = actual_l2_norm
                        last_budget_report["pre_projection_linf"] = float(np.max(np.abs(raw_cum_delta)))
                        last_budget_report["pre_projection_l2"] = float(np.sqrt(np.sum(raw_cum_delta ** 2)))
                        last_budget_report["epsilon_linf"] = eps_linf_norm
                        last_budget_report["epsilon_l2"] = eps_l2_norm
                    last_budget_report["linf_satisfied"] = bool(actual_linf_norm <= (eps_linf_norm or 0.0) + 1e-7) if eps_linf_norm is not None else True
                    last_budget_report["l2_satisfied"] = bool(actual_l2_norm <= (eps_l2_norm or 0.0) + 1e-7) if eps_l2_norm is not None else True
                    last_budget_report["all_constraints_satisfied"] = bool(last_budget_report["linf_satisfied"] and last_budget_report["l2_satisfied"])

                # Discrete intermediate representation for preview and metrics
                clipped_uint8 = normalized_float_to_image(current_norm)
                step_delta = clipped_uint8.astype(np.float64) - current_uint8.astype(np.float64)

                # Calculate intermediate metrics relative to base image
                diff_stats = calculate_norms_and_diff_stats(
                    clipped_uint8.astype(np.float64) - base_float,
                    l0_threshold=req.l0_threshold,
                )
                step_metrics = PerturbationMetrics(
                    mse=calculate_mse(base_rgb, clipped_uint8),
                    psnr=calculate_psnr(base_rgb, clipped_uint8),
                    ssim=calculate_ssim(base_rgb, clipped_uint8),
                    mae=calculate_mae(base_rgb, clipped_uint8),
                    l1_norm=diff_stats["l1_norm"],
                    l2_norm=diff_stats["l2_norm"],
                    linf_norm=diff_stats["linf_norm"],
                    max_abs_perturbation=diff_stats["max_abs_diff"],
                    min_difference=diff_stats["min_diff"],
                    max_difference=diff_stats["max_diff"],
                    mean_difference=diff_stats["mean_diff"],
                    std_diff=diff_stats["std_diff"],
                    l0_threshold=diff_stats["l0_threshold"],
                    changed_elements_count=diff_stats["changed_elements_count"],
                    total_elements_count=diff_stats["total_elements_count"],
                    changed_elements_percent=diff_stats["changed_elements_percent"],
                    changed_pixels_count=diff_stats["changed_pixels_count"],
                    total_pixels_count=diff_stats["total_pixels_count"],
                    changed_pixels_percent=diff_stats["changed_pixels_percent"],
                    modified_pixels_count=diff_stats["modified_pixels_count"],
                    modified_pixels_percent=diff_stats["modified_pixels_percent"],
                    lpips=None,
                    per_channel=None,
                    quantization_distortion=None,
                )

                step_results.append(
                    PerturbationStepResult(
                        step_index=step_idx,
                        step_id=op_id,
                        type=op.type,
                        name=display_name,
                        category=category,
                        enabled=True,
                        weight=op.weight,
                        formula_latex=formula_latex,
                        preview_data_url=array_to_base64_png(clipped_uint8, is_grayscale=False),
                        intermediate_metrics=step_metrics,
                    )
                )

                intermediate_rgb_list.append(clipped_uint8)
                individual_delta_list.append(step_delta)
                pipeline_steps_meta.append({
                    "step_index": step_idx,
                    "id": op_id,
                    "type": op.type,
                    "weight": op.weight,
                    "seed": op.seed,
                    "params": op.params,
                    "channels": op.channels,
                })

                # Advance Markov chain
                current_uint8 = clipped_uint8

            final_norm = current_norm
            final_float = current_norm * 255.0
            final_rgb = current_uint8
            budget_desc = f", with cumulative {b_type} budget enforcement" if b_type != "none" else ""
            overall_formula = r"I_{\text{final}} = P_n(\cdots P_2(P_1(I)))"
            overall_explanation = (
                f"Sequential composition of {len([s for s in step_results if s.enabled])} enabled operations{budget_desc}. "
                "Each operation was applied sequentially in Markov-chain order, enforcing cumulative global budget "
                "relative to the original base image at every step in normalized [0, 1] floating-point space."
            )

        # ==============================================================================
        # MODE 2: Weighted Perturbation Combination (Δ_combined = sum(w_k * Δ_k))
        # ==============================================================================
        elif req.composition_mode == "weighted":
            accumulated_delta_norm = np.zeros_like(base_norm)

            for idx, op in enumerate(operations):
                step_idx = idx + 1
                op_id = op.id or f"step-{step_idx}"

                active_channels = [channel_map[ch] for ch in op.channels if ch in channel_map]
                if not active_channels:
                    active_channels = [0, 1, 2]

                if not op.enabled:
                    step_metrics = PerturbationMetrics(
                        mse=0.0,
                        psnr=100.0,
                        ssim=1.0,
                        mae=0.0,
                        l1_norm=0.0,
                        l2_norm=0.0,
                        linf_norm=0.0,
                        max_abs_perturbation=0.0,
                        min_difference=0.0,
                        max_difference=0.0,
                        mean_difference=0.0,
                        std_diff=0.0,
                        l0_threshold=float(req.l0_threshold),
                        changed_elements_count=0,
                        total_elements_count=int(base_rgb.size),
                        changed_elements_percent=0.0,
                        changed_pixels_count=0,
                        total_pixels_count=int(base_rgb.shape[0] * base_rgb.shape[1]),
                        changed_pixels_percent=0.0,
                        modified_pixels_count=0,
                        modified_pixels_percent=0.0,
                        lpips=None,
                        per_channel=None,
                        quantization_distortion=None,
                    )
                    step_results.append(
                        PerturbationStepResult(
                            step_index=step_idx,
                            step_id=op_id,
                            type=op.type,
                            name=f"{op.type.capitalize()} (Disabled)",
                            category="bounded_noise",
                            enabled=False,
                            weight=op.weight,
                            formula_latex=r"w \cdot \Delta = 0",
                            preview_data_url=array_to_base64_png(base_rgb, is_grayscale=False),
                            intermediate_metrics=step_metrics,
                        )
                    )
                    intermediate_rgb_list.append(base_rgb)
                    individual_delta_list.append(np.zeros_like(base_float))
                    continue

                step_rng = np.random.default_rng(op.seed) if op.seed is not None else rng

                # In weighted mode, calculate each perturbation relative to the SAME base image in [0, 1]
                pert_norm, formula_latex, explanation, display_name, category = cls._apply_operator(
                    op_type=op.type,
                    input_float=base_norm,
                    active_channels=active_channels,
                    params=op.params,
                    rng=step_rng,
                    is_normalized=True,
                    parameter_unit=getattr(op, "parameter_unit", "pixel_0_255"),
                )

                # Individual delta relative to base image in normalized space
                raw_op_delta_norm = pert_norm - base_norm
                clamped_op_rgb = normalized_float_to_image(np.clip(base_norm + raw_op_delta_norm, 0.0, 1.0))

                # Scale delta by weight w_k in range [0, 5]
                weight_val = max(0.0, min(5.0, float(op.weight)))
                weighted_op_delta_norm = raw_op_delta_norm * weight_val
                accumulated_delta_norm += weighted_op_delta_norm

                diff_stats = calculate_norms_and_diff_stats(
                    clamped_op_rgb.astype(np.float64) - base_float,
                    l0_threshold=req.l0_threshold,
                )
                step_metrics = PerturbationMetrics(
                    mse=calculate_mse(base_rgb, clamped_op_rgb),
                    psnr=calculate_psnr(base_rgb, clamped_op_rgb),
                    ssim=calculate_ssim(base_rgb, clamped_op_rgb),
                    mae=calculate_mae(base_rgb, clamped_op_rgb),
                    l1_norm=diff_stats["l1_norm"],
                    l2_norm=diff_stats["l2_norm"],
                    linf_norm=diff_stats["linf_norm"],
                    max_abs_perturbation=diff_stats["max_abs_diff"],
                    min_difference=diff_stats["min_diff"],
                    max_difference=diff_stats["max_diff"],
                    mean_difference=diff_stats["mean_diff"],
                    std_diff=diff_stats["std_diff"],
                    l0_threshold=diff_stats["l0_threshold"],
                    changed_elements_count=diff_stats["changed_elements_count"],
                    total_elements_count=diff_stats["total_elements_count"],
                    changed_elements_percent=diff_stats["changed_elements_percent"],
                    changed_pixels_count=diff_stats["changed_pixels_count"],
                    total_pixels_count=diff_stats["total_pixels_count"],
                    changed_pixels_percent=diff_stats["changed_pixels_percent"],
                    modified_pixels_count=diff_stats["modified_pixels_count"],
                    modified_pixels_percent=diff_stats["modified_pixels_percent"],
                    lpips=None,
                    per_channel=None,
                    quantization_distortion=None,
                )

                step_results.append(
                    PerturbationStepResult(
                        step_index=step_idx,
                        step_id=op_id,
                        type=op.type,
                        name=f"{display_name} (w={weight_val:.2f})",
                        category=category,
                        enabled=True,
                        weight=weight_val,
                        formula_latex=f"{weight_val:.2f} \\cdot ({formula_latex})",
                        preview_data_url=array_to_base64_png(clamped_op_rgb, is_grayscale=False),
                        intermediate_metrics=step_metrics,
                    )
                )

                intermediate_rgb_list.append(clamped_op_rgb)
                individual_delta_list.append(weighted_op_delta_norm * 255.0)
                pipeline_steps_meta.append({
                    "step_index": step_idx,
                    "id": op_id,
                    "type": op.type,
                    "weight": weight_val,
                    "seed": op.seed,
                    "params": op.params,
                    "channels": op.channels,
                })

            # Apply global budget projection to combined normalized delta
            if b_type != "none":
                raw_comb_linf = float(np.max(np.abs(accumulated_delta_norm))) if accumulated_delta_norm.size > 0 else 0.0
                raw_comb_l2 = float(np.sqrt(np.sum(accumulated_delta_norm ** 2))) if accumulated_delta_norm.size > 0 else 0.0
                accumulated_delta_norm, last_budget_report = BudgetService.apply_budget(
                    accumulated_delta_norm,
                    budget_type=b_type,
                    epsilon_linf=eps_linf_norm,
                    epsilon_l2=eps_l2_norm,
                )
                bound_str = f", with {b_type} budget enforcement"
            else:
                raw_comb_linf = float(np.max(np.abs(accumulated_delta_norm))) if accumulated_delta_norm.size > 0 else 0.0
                raw_comb_l2 = float(np.sqrt(np.sum(accumulated_delta_norm ** 2))) if accumulated_delta_norm.size > 0 else 0.0
                bound_str = ""

            # Combine on base image and clip to [0, 1]
            final_norm = np.clip(base_norm + accumulated_delta_norm, 0.0, 1.0)
            actual_delta_norm = final_norm - base_norm
            actual_linf_norm = float(np.max(np.abs(actual_delta_norm))) if actual_delta_norm.size > 0 else 0.0
            actual_l2_norm = float(np.sqrt(np.sum(actual_delta_norm ** 2))) if actual_delta_norm.size > 0 else 0.0

            if last_budget_report is not None:
                if b_unit == "pixel_0_255":
                    last_budget_report["post_projection_linf"] = actual_linf_norm * 255.0
                    last_budget_report["post_projection_l2"] = actual_l2_norm * 255.0
                    last_budget_report["pre_projection_linf"] = raw_comb_linf * 255.0
                    last_budget_report["pre_projection_l2"] = raw_comb_l2 * 255.0
                    last_budget_report["epsilon_linf"] = eps_linf_px
                    last_budget_report["epsilon_l2"] = eps_l2_px
                else:
                    last_budget_report["post_projection_linf"] = actual_linf_norm
                    last_budget_report["post_projection_l2"] = actual_l2_norm
                    last_budget_report["pre_projection_linf"] = raw_comb_linf
                    last_budget_report["pre_projection_l2"] = raw_comb_l2
                    last_budget_report["epsilon_linf"] = eps_linf_norm
                    last_budget_report["epsilon_l2"] = eps_l2_norm
                last_budget_report["linf_satisfied"] = bool(actual_linf_norm <= (eps_linf_norm or 0.0) + 1e-7) if eps_linf_norm is not None else True
                last_budget_report["l2_satisfied"] = bool(actual_l2_norm <= (eps_l2_norm or 0.0) + 1e-7) if eps_l2_norm is not None else True
                last_budget_report["all_constraints_satisfied"] = bool(last_budget_report["linf_satisfied"] and last_budget_report["l2_satisfied"])

            final_float = final_norm * 255.0
            final_rgb = normalized_float_to_image(final_norm)

            overall_formula = r"\Delta_{\text{comb}} = \sum_{i=1}^n w_i \Delta_i, \quad I_{\text{final}} = \text{clip}(I + \Delta_{\text{comb}}, 0, 255)"
            overall_explanation = (
                f"Weighted superposition of {len([s for s in step_results if s.enabled])} perturbation fields "
                f"relative to the same base image{bound_str}."
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported composition mode: {req.composition_mode}")

        # Continuous vs uint8 quantization distortion E_quant = Δ_uint8 - Δ_continuous
        continuous_delta = final_float - base_float
        total_delta = final_rgb.astype(np.float64) - base_float
        final_gray = rgb_to_grayscale(final_rgb)

        quant_dist = calculate_quantization_distortion(final_float, final_rgb.astype(np.float64))
        quantization_distortion = {
            "mae": quant_dist["mae"],
            "mse": quant_dist["mse"],
            "rmse": quant_dist["rmse"],
            "max_error": quant_dist["max_error"],
            "linf": quant_dist["max_error"],  # backward-compatible alias
            "std": quant_dist["std"],
        }

        # Global final quality metrics
        final_mse = calculate_mse(base_rgb, final_rgb)
        final_psnr = calculate_psnr(base_rgb, final_rgb)
        final_ssim = calculate_ssim(base_rgb, final_rgb)
        final_mae = calculate_mae(base_rgb, final_rgb)
        final_diff_stats = calculate_norms_and_diff_stats(total_delta, l0_threshold=req.l0_threshold)

        # Per-channel metrics breakdown
        per_channel_metrics: Dict[str, ChannelMetrics] = {}
        if "per_channel" in final_diff_stats:
            for ch_name, ch_data in final_diff_stats["per_channel"].items():
                per_channel_metrics[ch_name] = ChannelMetrics(
                    channel=ch_name,
                    l1_norm=ch_data["l1_norm"],
                    l2_norm=ch_data["l2_norm"],
                    linf_norm=ch_data["linf_norm"],
                    mae=ch_data["mae"],
                    mse=ch_data["mse"],
                    mean_diff=ch_data["mean_diff"],
                    std_diff=ch_data["std_diff"],
                    min_diff=ch_data["min_diff"],
                    max_diff=ch_data["max_diff"],
                    changed_count=ch_data["changed_count"],
                )

        # LPIPS placeholder hook for Phase 5 deep feature perceptual distance
        lpips_value = calculate_lpips_hook(base_rgb, final_rgb)

        metrics = PerturbationMetrics(
            mse=final_mse,
            psnr=final_psnr,
            ssim=final_ssim,
            mae=final_mae,
            l1_norm=final_diff_stats["l1_norm"],
            l2_norm=final_diff_stats["l2_norm"],
            linf_norm=final_diff_stats["linf_norm"],
            max_abs_perturbation=final_diff_stats["max_abs_diff"],
            min_difference=final_diff_stats["min_diff"],
            max_difference=final_diff_stats["max_diff"],
            mean_difference=final_diff_stats["mean_diff"],
            std_diff=final_diff_stats["std_diff"],
            l0_threshold=final_diff_stats["l0_threshold"],
            changed_elements_count=final_diff_stats["changed_elements_count"],
            total_elements_count=final_diff_stats["total_elements_count"],
            changed_elements_percent=final_diff_stats["changed_elements_percent"],
            changed_pixels_count=final_diff_stats["changed_pixels_count"],
            total_pixels_count=final_diff_stats["total_pixels_count"],
            changed_pixels_percent=final_diff_stats["changed_pixels_percent"],
            modified_pixels_count=final_diff_stats["modified_pixels_count"],
            modified_pixels_percent=final_diff_stats["modified_pixels_percent"],
            lpips=lpips_value,
            per_channel=per_channel_metrics,
            quantization_distortion=quantization_distortion,
        )

        # Previews
        original_data_url = array_to_base64_png(base_rgb, is_grayscale=False)
        perturbed_rgb_data_url = array_to_base64_png(final_rgb, is_grayscale=False)
        perturbed_gray_data_url = array_to_base64_png(final_gray, is_grayscale=True)
        heatmap_data_url = generate_error_heatmap_base64(base_rgb, final_rgb)

        # Signed difference visualization
        signed_vis = np.clip((total_delta + 255.0) / 2.0, 0, 255).astype(np.uint8)
        signed_diff_data_url = array_to_base64_png(signed_vis, is_grayscale=False)

        # Localized Multi-Step Pixel Inspection
        pixel_inspection: Optional[PerturbationPixelInspection] = None
        if req.selected_point is not None:
            px, py = req.selected_point
            if 0 <= px < w and 0 <= py < h:
                r_rad = max(1, min(10, req.submatrix_radius))
                y_min = max(0, py - r_rad)
                y_max = min(h - 1, py + r_rad)
                x_min = max(0, px - r_rad)
                x_max = min(w - 1, px + r_rad)

                orig_px = [int(base_rgb[py, px, c_i]) for c_i in range(3)]
                final_px = [int(final_rgb[py, px, c_i]) for c_i in range(3)]
                delta_px = [float(final_px[c_i] - orig_px[c_i]) for c_i in range(3)]
                abs_delta_px = [abs(d) for d in delta_px]

                orig_g = float(base_gray[py, px])
                final_g = float(final_gray[py, px])
                delta_g = float(final_g - orig_g)

                # Step-by-step intermediate pixel tracking at (px, py)
                inter_pixels: List[IntermediatePixelValue] = []
                prev_rgb = orig_px
                for s_res, s_img in zip(step_results, intermediate_rgb_list):
                    cur_rgb = [int(s_img[py, px, c_i]) for c_i in range(3)]
                    delta_prev = [float(cur_rgb[c_i] - prev_rgb[c_i]) for c_i in range(3)]
                    delta_orig = [float(cur_rgb[c_i] - orig_px[c_i]) for c_i in range(3)]
                    inter_pixels.append(
                        IntermediatePixelValue(
                            step_index=s_res.step_index,
                            step_name=s_res.name,
                            rgb=cur_rgb,
                            delta_from_previous=delta_prev,
                            delta_from_original=delta_orig,
                        )
                    )
                    prev_rgb = cur_rgb

                sub_delta_r = total_delta[y_min : y_max + 1, x_min : x_max + 1, 0].tolist()
                sub_delta_g = total_delta[y_min : y_max + 1, x_min : x_max + 1, 1].tolist()
                sub_delta_b = total_delta[y_min : y_max + 1, x_min : x_max + 1, 2].tolist()
                sub_delta_gray = (
                    final_gray[y_min : y_max + 1, x_min : x_max + 1].astype(np.float64)
                    - base_gray[y_min : y_max + 1, x_min : x_max + 1].astype(np.float64)
                ).tolist()

                pixel_inspection = PerturbationPixelInspection(
                    x=px,
                    y=py,
                    original_rgb=orig_px,
                    intermediate_pixels=inter_pixels,
                    perturbed_rgb=final_px,
                    delta_rgb=delta_px,
                    abs_delta_rgb=abs_delta_px,
                    original_gray=orig_g,
                    perturbed_gray=final_g,
                    delta_gray=delta_g,
                    submatrix_radius=r_rad,
                    submatrix_bounds={"ymin": y_min, "ymax": y_max, "xmin": x_min, "xmax": x_max},
                    submatrix_delta_r=sub_delta_r,
                    submatrix_delta_g=sub_delta_g,
                    submatrix_delta_b=sub_delta_b,
                    submatrix_delta_gray=sub_delta_gray,
                )

        # Cache in StoredImage for persistence and lossless download
        method_name = "pipeline" if len(operations) > 1 else (operations[0].type if operations else "none")
        pert_metadata = {
            "method": method_name,
            "composition_mode": req.composition_mode,
            "budget": req.budget.model_dump() if req.budget else None,
            "budget_unit": b_unit,
            "global_bound": eps_linf_px if (b_type in ["linf", "both"] and eps_linf_px is not None) else req.global_bound,
            "base_source": req.base_source,
            "seed": req.seed,
            "metrics": metrics.model_dump(),
            "operations_count": len(operations),
            "enabled_count": len([op for op in operations if op.enabled]),
            "budget_report": last_budget_report,
            "input_representation": "normalized_float64_[0,1]",
            "output_representation": "uint8_image_[0,255]_and_normalized_float64_[0,1]",
            "quantization_distortion": quantization_distortion,
        }

        actual_delta_norm = final_norm - base_norm

        stored.set_perturbed(
            rgb=final_rgb,
            gray=final_gray,
            delta=total_delta,
            meta=pert_metadata,
            intermediate_rgbs=intermediate_rgb_list,
            individual_deltas=individual_delta_list,
            pipeline_steps_meta=pipeline_steps_meta,
            perturbed_rgb_normalized=final_norm,
            perturbation_delta_normalized=actual_delta_norm,
        )

        params_dict = {
            "composition_mode": req.composition_mode,
            "budget": req.budget.model_dump() if req.budget else None,
            "global_bound": req.global_bound,
            "base_source": req.base_source,
            "operations": [op.model_dump() for op in operations],
            "seed": req.seed,
        }

        effective_global_bound = eps_linf_px if (b_type in ["linf", "both"] and eps_linf_px is not None) else req.global_bound

        return PerturbationResponse(
            image_id=stored.image_id,
            method=method_name,
            composition_mode=req.composition_mode,
            base_source=req.base_source,
            width=w,
            height=h,
            channels=c,
            global_bound=effective_global_bound,
            budget_report=last_budget_report,
            parameters=params_dict,
            reproducibility_seed=req.seed,
            original_data_url=original_data_url,
            perturbed_rgb_data_url=perturbed_rgb_data_url,
            perturbed_grayscale_data_url=perturbed_gray_data_url,
            difference_heatmap_data_url=heatmap_data_url,
            signed_difference_data_url=signed_diff_data_url,
            metrics=metrics,
            pixel_inspection=pixel_inspection,
            intermediate_steps=step_results,
            method_explanation=overall_explanation,
            mathematical_formula=overall_formula,
        )


# Global singleton instance
perturbation_service = PerturbationService()
