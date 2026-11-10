"""Transformation service implementing mathematical image transformations and affine composition."""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from fastapi import HTTPException
from app.services.image_service import StoredImage
from app.utils.image_utils import array_to_base64_png
from app.utils.math_utils import rgb_to_grayscale
from app.models.schemas import (
    TransformationStep,
    TransformationRequest,
    TransformationResponse,
    MatrixStepInfo,
)


class TransformationService:
    """Mathematical transformation engine handling 2D affine operations in homogeneous coordinates."""

    @staticmethod
    def get_center_matrices(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns forward and inverse translation matrices to/from the image center."""
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        t_to_origin = np.array([
            [1.0, 0.0, -cx],
            [0.0, 1.0, -cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        t_back = np.array([
            [1.0, 0.0, cx],
            [0.0, 1.0, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return t_to_origin, t_back

    @classmethod
    def build_step_matrix(
        cls, step: TransformationStep, w: int, h: int
    ) -> Tuple[np.ndarray, MatrixStepInfo]:
        """
        Constructs the 3x3 homogeneous matrix for a single transformation step.
        Centers rotation, scaling, reflection, and shearing around the image center.
        """
        op = step.type
        params = step.params
        t_to_origin, t_back = cls.get_center_matrices(w, h)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        if op == "rotate":
            angle_deg = float(params.get("angle", 0.0))
            theta = np.deg2rad(angle_deg)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            r_origin = np.array([
                [cos_t, -sin_t, 0.0],
                [sin_t, cos_t, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            m_3x3 = t_back @ r_origin @ t_to_origin

            formula = r"R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}"
            desc = (
                f"Rotation by {angle_deg:.1f}° around image center ({cx:.1f}, {cy:.1f}). "
                f"Coordinates rotate counter-clockwise (in Cartesian space) or clockwise in matrix image axes."
            )
            step_info = MatrixStepInfo(
                name=f"Rotation ({angle_deg:.1f}°)",
                operation_type="rotate",
                matrix_3x3=np.round(m_3x3, 4).tolist(),
                formula_latex=formula,
                description=desc,
            )
            return m_3x3, step_info

        elif op == "scale":
            sx = float(params.get("sx", 1.0))
            sy = float(params.get("sy", 1.0))

            s_origin = np.array([
                [sx, 0.0, 0.0],
                [0.0, sy, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            m_3x3 = t_back @ s_origin @ t_to_origin

            formula = r"S(s_x, s_y) = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}"
            desc = (
                f"Scaling by factor s_x = {sx:.2f}, s_y = {sy:.2f} relative to image center ({cx:.1f}, {cy:.1f}). "
                f"Factors > 1 enlarge coordinates, factors < 1 compress coordinates."
            )
            step_info = MatrixStepInfo(
                name=f"Scale (sx={sx:.2f}, sy={sy:.2f})",
                operation_type="scale",
                matrix_3x3=np.round(m_3x3, 4).tolist(),
                formula_latex=formula,
                description=desc,
            )
            return m_3x3, step_info

        elif op == "reflect":
            axis = str(params.get("axis", "horizontal")).lower()
            if axis == "horizontal":
                # Reflection across vertical axis x = cx (flips horizontal coordinates left-right)
                f_origin = np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                formula = r"F_{horizontal} = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}"
                desc = f"Horizontal reflection (flip left-right across vertical centerline x = {cx:.1f})."
                name = "Horizontal Reflection"
            elif axis == "vertical":
                # Reflection across horizontal axis y = cy (flips vertical coordinates top-down)
                f_origin = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                formula = r"F_{vertical} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}"
                desc = f"Vertical reflection (flip top-bottom across horizontal centerline y = {cy:.1f})."
                name = "Vertical Reflection"
            else:  # "both" (point reflection around center)
                f_origin = np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                formula = r"F_{both} = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}"
                desc = f"Point reflection (inversion across center point ({cx:.1f}, {cy:.1f}))."
                name = "Both Axes Reflection"

            m_3x3 = t_back @ f_origin @ t_to_origin
            step_info = MatrixStepInfo(
                name=name,
                operation_type="reflect",
                matrix_3x3=np.round(m_3x3, 4).tolist(),
                formula_latex=formula,
                description=desc,
            )
            return m_3x3, step_info

        elif op == "shear":
            kx = float(params.get("kx", 0.0))
            ky = float(params.get("ky", 0.0))

            h_origin = np.array([
                [1.0, kx, 0.0],
                [ky, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            m_3x3 = t_back @ h_origin @ t_to_origin

            formula = r"H(k_x, k_y) = \begin{bmatrix} 1 & k_x \\ k_y & 1 \end{bmatrix}"
            desc = (
                f"Shear mapping with horizontal factor k_x = {kx:.2f} and vertical factor k_y = {ky:.2f}. "
                f"Shifts coordinate lines parallel to axes proportionally to distance from center."
            )
            step_info = MatrixStepInfo(
                name=f"Shear (kx={kx:.2f}, ky={ky:.2f})",
                operation_type="shear",
                matrix_3x3=np.round(m_3x3, 4).tolist(),
                formula_latex=formula,
                description=desc,
            )
            return m_3x3, step_info

        elif op == "translate":
            tx = float(params.get("tx", 0.0))
            ty = float(params.get("ty", 0.0))

            m_3x3 = np.array([
                [1.0, 0.0, tx],
                [0.0, 1.0, ty],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            formula = r"T(t_x, t_y) = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}"
            desc = (
                f"Rigid translation displacing coordinates by Δx = {tx:.1f} and Δy = {ty:.1f}. "
                f"Requires homogeneous coordinates [x, y, 1]^T to express non-origin translations via matrix multiplication."
            )
            step_info = MatrixStepInfo(
                name=f"Translation (tx={tx:.1f}, ty={ty:.1f})",
                operation_type="translate",
                matrix_3x3=np.round(m_3x3, 4).tolist(),
                formula_latex=formula,
                description=desc,
            )
            return m_3x3, step_info

        elif op == "random_affine":
            seed_val = params.get("seed", 42)
            rng = np.random.default_rng(seed_val) if seed_val is not None else np.random.default_rng()
            max_rot = float(params.get("max_rotation_deg", 15.0))
            max_scale = float(params.get("max_scale_delta", 0.1))
            max_trans = float(params.get("max_translation_px", 5.0))

            rot_deg = float(rng.uniform(-max_rot, max_rot))
            scale_factor = float(rng.uniform(1.0 - max_scale, 1.0 + max_scale))
            tx = float(rng.uniform(-max_trans, max_trans))
            ty = float(rng.uniform(-max_trans, max_trans))

            theta = np.deg2rad(rot_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            r_s = np.array([
                [scale_factor * cos_t, -scale_factor * sin_t, 0.0],
                [scale_factor * sin_t, scale_factor * cos_t, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            t_m = np.array([
                [1.0, 0.0, tx],
                [0.0, 1.0, ty],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            m_3x3 = t_back @ t_m @ r_s @ t_to_origin

            formula = r"M_{\text{random}} = T(\Delta x, \Delta y) \cdot S(s) \cdot R(\theta)"
            desc = (
                f"Seeded random affine (seed={seed_val}): rotation {rot_deg:.1f}°, scale {scale_factor:.2f}x, "
                f"translation ({tx:.1f}, {ty:.1f})px. Prepares pipeline for Phase 5 EOT."
            )
            step_info = MatrixStepInfo(
                name=f"Random Affine (seed={seed_val})",
                operation_type="random_affine",
                matrix_3x3=np.round(m_3x3, 4).tolist(),
                formula_latex=formula,
                description=desc,
            )
            return m_3x3, step_info

        elif op == "resize":
            target_w = int(params.get("target_width", w))
            target_h = int(params.get("target_height", h))
            sx = target_w / max(1.0, float(w))
            sy = target_h / max(1.0, float(h))
            m_3x3 = np.array([
                [sx, 0.0, 0.0],
                [0.0, sy, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            formula = r"\text{Resize}(W_{\text{target}}, H_{\text{target}})"
            desc = f"Spatial resample from ({w}x{h}) to ({target_w}x{target_h}). Simulates spatial scaling resolution loss."
            step_info = MatrixStepInfo(
                name=f"Resize ({target_w}x{target_h})",
                operation_type="resize",
                matrix_3x3=np.round(m_3x3, 4).tolist(),
                formula_latex=formula,
                description=desc,
            )
            return m_3x3, step_info

        elif op == "jpeg_compression":
            quality = int(params.get("quality", 75))
            m_3x3 = np.eye(3, dtype=np.float64)
            formula = r"\text{JPEG}(Q)"
            desc = (
                f"JPEG compression/decompression simulation at quality Q={quality}. "
                "Note: JPEG DCT quantization is non-differentiable; in Phase 5 EOT optimization, "
                "Backward Pass Differentiable Approximation (BPDA) will be used."
            )
            step_info = MatrixStepInfo(
                name=f"JPEG Compression (Q={quality})",
                operation_type="jpeg_compression",
                matrix_3x3=np.round(m_3x3, 4).tolist(),
                formula_latex=formula,
                description=desc,
            )
            return m_3x3, step_info

        else:
            raise ValueError(f"Unknown transformation operation: '{op}'")

    @classmethod
    def apply_transformations(
        cls, stored: StoredImage, req: TransformationRequest
    ) -> TransformationResponse:
        """
        Executes sequence of transformations on the stored image, preserving original image arrays.
        Supports bidirectional source: 'original' or 'perturbed' (Phase 4).
        Computes composite affine matrix, warps RGB and grayscale tensors, and caches transformed arrays.
        """
        # Bidirectional pipeline support: transform original or perturbed image
        base_source = getattr(req, "base_source", "original")
        if base_source == "perturbed":
            if stored.perturbed_rgb_array is None:
                raise HTTPException(
                    status_code=400,
                    detail="No Phase 4 perturbed image found. Please apply a perturbation first or select 'original' as base source."
                )
            source_rgb = stored.perturbed_rgb_array.copy()
        else:
            source_rgb = stored.rgb_array.copy()

        h, w = source_rgb.shape[:2]
        steps_info: List[MatrixStepInfo] = []
        composite_m = np.eye(3, dtype=np.float64)

        # Composite transformation is applied sequentially: M_composite = M_n * ... * M_2 * M_1
        for step in req.operations:
            step_m, step_info = cls.build_step_matrix(step, w, h)
            composite_m = step_m @ composite_m
            steps_info.append(step_info)

        det = float(np.linalg.det(composite_m))
        is_invertible = abs(det) > 1e-7

        # Determine canvas dimensions and bounding box adjustments
        if req.canvas_mode == "fit":
            # Transform all 4 image corners: (0,0), (w-1,0), (0,h-1), (w-1,h-1)
            corners = np.array([
                [0.0, w - 1.0, 0.0, w - 1.0],
                [0.0, 0.0, h - 1.0, h - 1.0],
                [1.0, 1.0, 1.0, 1.0]
            ], dtype=np.float64)
            transformed_corners = composite_m @ corners

            min_x = float(transformed_corners[0].min())
            max_x = float(transformed_corners[0].max())
            min_y = float(transformed_corners[1].min())
            max_y = float(transformed_corners[1].max())

            out_w = max(1, int(np.ceil(max_x - min_x + 1)))
            out_h = max(1, int(np.ceil(max_y - min_y + 1)))

            # Shift transformed coordinates so (min_x, min_y) lands at (0, 0) in new canvas
            t_shift = np.array([
                [1.0, 0.0, -min_x],
                [0.0, 1.0, -min_y],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            effective_m = t_shift @ composite_m
        else:
            # "same" mode preserves original image width and height
            out_w, out_h = w, h
            effective_m = composite_m

        # Set up interpolation
        if req.interpolation == "nearest":
            interp_flag = cv2.INTER_NEAREST
        elif req.interpolation == "bicubic":
            interp_flag = cv2.INTER_CUBIC
        else:
            interp_flag = cv2.INTER_LINEAR

        # Affine warp using OpenCV's backward mapping
        affine_2x3 = effective_m[:2, :].astype(np.float32)
        transformed_rgb = cv2.warpAffine(
            source_rgb,
            affine_2x3,
            (out_w, out_h),
            flags=interp_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Apply JPEG compression round-trip simulation if present in operations
        for op in req.operations:
            if op.type == "jpeg_compression":
                quality = int(op.params.get("quality", 75))
                success, enc_buf = cv2.imencode(".jpg", transformed_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if success:
                    transformed_rgb = cv2.imdecode(enc_buf, cv2.IMREAD_COLOR)
        transformed_gray = rgb_to_grayscale(transformed_rgb)

        # Cache transformed arrays on StoredImage for subsequent inspection, download, or analysis
        stored.set_transformed(
            rgb=transformed_rgb,
            gray=transformed_gray,
            meta={
                "operations_count": len(req.operations),
                "canvas_mode": req.canvas_mode,
                "interpolation": req.interpolation,
                "composite_matrix": effective_m.tolist(),
            }
        )

        # Generate base64 previews for browser presentation
        trans_rgb_url = array_to_base64_png(transformed_rgb, is_grayscale=False)
        trans_gray_url = array_to_base64_png(transformed_gray, is_grayscale=True)

        # Map optional user point
        orig_pt = None
        trans_pt = None
        if req.selected_point is not None:
            px, py = req.selected_point
            orig_pt = (round(float(px), 2), round(float(py), 2))
            pt_vec = np.array([px, py, 1.0], dtype=np.float64)
            pt_trans = effective_m @ pt_vec
            trans_pt = (round(float(pt_trans[0]), 2), round(float(pt_trans[1]), 2))

        multiplication_explanation = (
            "Composite transformation matrix calculation: M_composite = M_n × M_{n-1} × ... × M_1. "
            "Matrices multiply from right to left in operator notation: the first operation applied to a coordinate "
            "vector p is M_1, followed sequentially by each subsequent operation. Because matrix multiplication "
            "is generally non-commutative (AB ≠ BA), altering the sequence order fundamentally changes the resulting "
            "coordinate mapping and output image."
        )

        return TransformationResponse(
            original_width=w,
            original_height=h,
            transformed_width=out_w,
            transformed_height=out_h,
            composite_matrix_3x3=np.round(effective_m, 4).tolist(),
            steps=steps_info,
            matrix_multiplication_explanation=multiplication_explanation,
            original_point=orig_pt,
            transformed_point=trans_pt,
            transformed_rgb_data_url=trans_rgb_url,
            transformed_grayscale_data_url=trans_gray_url,
            interpolation=req.interpolation,
            canvas_mode=req.canvas_mode,
            determinant=round(det, 4),
            is_invertible=is_invertible,
        )


transformation_service = TransformationService()

