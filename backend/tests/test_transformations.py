"""Unit and integration tests for Phase 3: Mathematical Image Transformations."""

import pytest
import numpy as np
from app.services.image_service import image_service
from app.services.transformation_service import transformation_service
from app.api.routes import load_sample_image, transform_image
from app.models.schemas import TransformationStep, TransformationRequest


def test_rotation_angles():
    """Verify 90, 180, 270, and 360 degree rotations around the image center."""
    meta = load_sample_image()
    stored = image_service.get_image(meta.image_id)
    w, h = stored.width, stored.height
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # 1. 360-degree rotation must equal identity matrix
    m_360, info_360 = transformation_service.build_step_matrix(
        TransformationStep(type="rotate", params={"angle": 360.0}), w, h
    )
    assert np.allclose(m_360, np.eye(3), atol=1e-5), "360 deg rotation must be identity matrix"

    # 2. 90-degree rotation
    m_90, info_90 = transformation_service.build_step_matrix(
        TransformationStep(type="rotate", params={"angle": 90.0}), w, h
    )
    # Rotating center point must leave center invariant
    center_pt = np.array([cx, cy, 1.0])
    center_rotated = m_90 @ center_pt
    assert np.allclose(center_rotated[:2], [cx, cy], atol=1e-5), "Rotation around center must preserve center coordinates"

    # 3. Four 90-degree rotations composed must equal identity
    m_90_x4 = m_90 @ m_90 @ m_90 @ m_90
    assert np.allclose(m_90_x4, np.eye(3), atol=1e-5), "4x 90 deg rotation must equal identity"

    # 4. Positive and negative angles: R(45) * R(-45) = I
    m_p45, _ = transformation_service.build_step_matrix(TransformationStep(type="rotate", params={"angle": 45.0}), w, h)
    m_n45, _ = transformation_service.build_step_matrix(TransformationStep(type="rotate", params={"angle": -45.0}), w, h)
    assert np.allclose(m_p45 @ m_n45, np.eye(3), atol=1e-5), "R(45) * R(-45) must equal identity"


def test_scaling_factors():
    """Verify uniform and independent X/Y scaling."""
    meta = load_sample_image()
    stored = image_service.get_image(meta.image_id)
    w, h = stored.width, stored.height
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # 1. Scale by 1.0 must be identity
    m_s1, _ = transformation_service.build_step_matrix(
        TransformationStep(type="scale", params={"sx": 1.0, "sy": 1.0}), w, h
    )
    assert np.allclose(m_s1, np.eye(3), atol=1e-5)

    # 2. Scale by 2.0 and then 0.5 must restore original coordinates
    m_s2, _ = transformation_service.build_step_matrix(
        TransformationStep(type="scale", params={"sx": 2.0, "sy": 2.0}), w, h
    )
    m_s05, _ = transformation_service.build_step_matrix(
        TransformationStep(type="scale", params={"sx": 0.5, "sy": 0.5}), w, h
    )
    assert np.allclose(m_s2 @ m_s05, np.eye(3), atol=1e-5), "S(2) * S(0.5) must equal identity"

    # Center invariant under scaling around center
    center_scaled = m_s2 @ np.array([cx, cy, 1.0])
    assert np.allclose(center_scaled[:2], [cx, cy], atol=1e-5)


def test_reflection_properties():
    """Verify horizontal, vertical, and both axes reflections (involutions: F * F = I)."""
    meta = load_sample_image()
    stored = image_service.get_image(meta.image_id)
    w, h = stored.width, stored.height

    # Horizontal reflection twice = Identity
    m_fh, _ = transformation_service.build_step_matrix(
        TransformationStep(type="reflect", params={"axis": "horizontal"}), w, h
    )
    assert np.allclose(m_fh @ m_fh, np.eye(3), atol=1e-5), "F_h * F_h must equal identity"

    # Vertical reflection twice = Identity
    m_fv, _ = transformation_service.build_step_matrix(
        TransformationStep(type="reflect", params={"axis": "vertical"}), w, h
    )
    assert np.allclose(m_fv @ m_fv, np.eye(3), atol=1e-5), "F_v * F_v must equal identity"

    # Both axes reflection twice = Identity
    m_fb, _ = transformation_service.build_step_matrix(
        TransformationStep(type="reflect", params={"axis": "both"}), w, h
    )
    assert np.allclose(m_fb @ m_fb, np.eye(3), atol=1e-5), "F_both * F_both must equal identity"


def test_shearing_and_determinant():
    """Verify shearing matrices and area preservation (det = 1 - kx*ky)."""
    meta = load_sample_image()
    stored = image_service.get_image(meta.image_id)
    w, h = stored.width, stored.height

    # Pure X-shear (kx=0.5, ky=0) -> det = 1 (area preserving)
    m_hx, _ = transformation_service.build_step_matrix(
        TransformationStep(type="shear", params={"kx": 0.5, "ky": 0.0}), w, h
    )
    assert np.isclose(np.linalg.det(m_hx), 1.0, atol=1e-5)

    # Pure Y-shear (kx=0, ky=0.5) -> det = 1 (area preserving)
    m_hy, _ = transformation_service.build_step_matrix(
        TransformationStep(type="shear", params={"kx": 0.0, "ky": 0.5}), w, h
    )
    assert np.isclose(np.linalg.det(m_hy), 1.0, atol=1e-5)


def test_translation_homogeneous_coordinates():
    """Verify translation in homogeneous coordinates and inverse translation."""
    w, h = 100, 100
    m_t, info = transformation_service.build_step_matrix(
        TransformationStep(type="translate", params={"tx": 25.0, "ty": -15.0}), w, h
    )
    assert m_t[0, 2] == 25.0
    assert m_t[1, 2] == -15.0
    assert m_t[2, 2] == 1.0

    # Inverse translation
    m_t_inv, _ = transformation_service.build_step_matrix(
        TransformationStep(type="translate", params={"tx": -25.0, "ty": 15.0}), w, h
    )
    assert np.allclose(m_t @ m_t_inv, np.eye(3), atol=1e-5)


def test_transformation_composition_non_commutative():
    """Verify that transformation composition is non-commutative: A * B != B * A."""
    meta = load_sample_image()
    stored = image_service.get_image(meta.image_id)
    w, h = stored.width, stored.height

    # Operation 1: Rotate 90
    m_rot, _ = transformation_service.build_step_matrix(
        TransformationStep(type="rotate", params={"angle": 90.0}), w, h
    )
    # Operation 2: Translate tx=50, ty=0
    m_trans, _ = transformation_service.build_step_matrix(
        TransformationStep(type="translate", params={"tx": 50.0, "ty": 0.0}), w, h
    )

    # Rotate THEN Translate: M_trans * M_rot
    order_a = m_trans @ m_rot
    # Translate THEN Rotate: M_rot * M_trans
    order_b = m_rot @ m_trans

    assert not np.allclose(order_a, order_b), "Matrix multiplication must be non-commutative (AB != BA)"


def test_full_transformation_pipeline_execution():
    """Verify API pipeline execution of composite transformations on the sample image."""
    meta = load_sample_image()
    req = TransformationRequest(
        operations=[
            TransformationStep(type="rotate", params={"angle": 45.0}),
            TransformationStep(type="scale", params={"sx": 1.2, "sy": 1.2}),
            TransformationStep(type="translate", params={"tx": 10.0, "ty": -10.0}),
        ],
        interpolation="bilinear",
        canvas_mode="fit",
        selected_point=(128.0, 128.0),
    )

    res = transform_image(meta.image_id, req)

    assert res.original_width == 256
    assert res.original_height == 256
    assert res.transformed_width > 0
    assert res.transformed_height > 0
    assert len(res.steps) == 3
    assert res.steps[0].operation_type == "rotate"
    assert res.steps[1].operation_type == "scale"
    assert res.steps[2].operation_type == "translate"
    assert res.transformed_point is not None
    assert res.transformed_rgb_data_url.startswith("data:image/png;base64,")
    assert res.transformed_grayscale_data_url.startswith("data:image/png;base64,")

    # Verify original image in storage was NOT mutated
    stored = image_service.get_image(meta.image_id)
    assert stored.rgb_array.shape == (256, 256, 3)
    assert stored.transformed_rgb_array is not None

