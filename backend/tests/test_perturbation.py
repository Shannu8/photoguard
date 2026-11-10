"""
Phase 4 Backend Tests: Mathematical Image Perturbation Engine.
Verifies bounded noise, reproducibility, tensor immutability, clipping,
metric calculations (MSE, PSNR, SSIM, MAE, Norms), and tensor downloads.
"""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.image_service import image_service
from app.models.schemas import PerturbationRequest

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_image_store():
    """Ensure in-memory store is clean before each test."""
    image_service._store.clear()
    yield
    image_service._store.clear()


def create_test_image(width: int = 32, height: int = 32, color=(100, 150, 200)) -> str:
    """Helper creating a test image and returning its image_id."""
    img = Image.new("RGB", (width, height), color=color)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)

    res = client.post(
        "/api/upload",
        files={"file": ("test_patch.png", bio.getvalue(), "image/png")},
    )
    assert res.status_code == 200
    return res.json()["image_id"]


def test_bounded_additive_perturbation_norm():
    """Verify that bounded additive perturbation strictly satisfies ||Δ||_∞ <= ε."""
    img_id = create_test_image(32, 32, (120, 120, 120))
    eps = 15.0

    req_data = {
        "method": "additive",
        "base_source": "original",
        "channels": ["R", "G", "B"],
        "epsilon": eps,
        "additive_mode": "random",
        "seed": 12345,
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req_data)
    assert res.status_code == 200
    data = res.json()

    metrics = data["metrics"]
    assert metrics["linf_norm"] <= eps + 1e-5
    assert metrics["max_abs_perturbation"] <= eps + 1e-5
    assert metrics["mse"] > 0
    assert metrics["mae"] > 0
    assert 0.0 <= metrics["ssim"] <= 1.0


def test_deterministic_reproducibility_via_seed():
    """Verify that using the same random seed produces bit-identical perturbed tensors."""
    img_id = create_test_image(32, 32, (100, 150, 200))

    req = {
        "method": "gaussian",
        "base_source": "original",
        "channels": ["R", "G", "B"],
        "sigma": 20.0,
        "seed": 9999,
    }
    res1 = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res1.status_code == 200
    data1 = res1.json()

    res2 = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res2.status_code == 200
    data2 = res2.json()

    assert data1["perturbed_rgb_data_url"] == data2["perturbed_rgb_data_url"]
    assert data1["metrics"]["mse"] == pytest.approx(data2["metrics"]["mse"], rel=1e-5)
    assert data1["metrics"]["ssim"] == pytest.approx(data2["metrics"]["ssim"], rel=1e-5)


def test_channel_selective_perturbation():
    """Verify that selecting only channel R leaves channels G and B completely unaltered."""
    img_id = create_test_image(32, 32, (80, 140, 210))

    req = {
        "method": "additive",
        "base_source": "original",
        "channels": ["R"],
        "epsilon": 25.0,
        "additive_mode": "positive_const",
        "seed": 42,
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200

    stored = image_service.get_image(img_id)
    orig = stored.rgb_array
    pert = stored.perturbed_rgb_array

    # R channel should be shifted by +25 (80 + 25 = 105)
    assert np.all(pert[:, :, 0] == 105)
    # G and B channels must remain bitwise identical
    assert np.array_equal(pert[:, :, 1], orig[:, :, 1])
    assert np.array_equal(pert[:, :, 2], orig[:, :, 2])


def test_strict_clipping_bounds():
    """Verify that perturbations never produce underflow (<0) or overflow (>255)."""
    # Create an image near boundaries (250, 5, 255)
    img_id = create_test_image(16, 16, (250, 5, 255))

    req = {
        "method": "additive",
        "base_source": "original",
        "channels": ["R", "G", "B"],
        "epsilon": 50.0,
        "additive_mode": "positive_const",
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200

    stored = image_service.get_image(img_id)
    pert = stored.perturbed_rgb_array
    assert pert.dtype == np.uint8
    assert np.max(pert) <= 255
    assert np.min(pert) >= 0
    # 250 + 50 should be clipped to 255
    assert np.all(pert[:, :, 0] == 255)
    assert np.all(pert[:, :, 2] == 255)


def test_pixel_inversion():
    """Verify pixel inversion produces exact 255 - I photographic negative."""
    img_id = create_test_image(16, 16, (50, 100, 150))

    req = {
        "method": "inversion",
        "base_source": "original",
        "channels": ["R", "G", "B"],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200

    stored = image_service.get_image(img_id)
    pert = stored.perturbed_rgb_array
    assert np.all(pert[:, :, 0] == 255 - 50)
    assert np.all(pert[:, :, 1] == 255 - 100)
    assert np.all(pert[:, :, 2] == 255 - 150)


def test_salt_and_pepper_values():
    """Verify salt and pepper noise sets corrupted pixels strictly to 0 or 255."""
    img_id = create_test_image(32, 32, (128, 128, 128))

    req = {
        "method": "salt_and_pepper",
        "base_source": "original",
        "channels": ["R", "G", "B"],
        "salt_pepper_prob": 0.3,
        "salt_vs_pepper_ratio": 0.5,
        "seed": 777,
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200

    stored = image_service.get_image(img_id)
    pert = stored.perturbed_rgb_array
    # All pixels should either be uncorrupted (128) or salt (255) or pepper (0)
    unique_vals = np.unique(pert)
    for val in unique_vals:
        assert val in [0, 128, 255]


def test_multiplicative_perturbation():
    """Verify multiplicative perturbation scales pixel intensities proportionally."""
    img_id = create_test_image(16, 16, (100, 100, 100))

    req = {
        "method": "multiplicative",
        "base_source": "original",
        "channels": ["R", "G", "B"],
        "multiplicative_factor": 0.2,
        "seed": 42,
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200

    stored = image_service.get_image(img_id)
    pert = stored.perturbed_rgb_array
    # Since alpha = 0.2, values must stay within [100*(1-0.2), 100*(1+0.2)] = [80, 120]
    assert np.min(pert) >= 80 - 1
    assert np.max(pert) <= 120 + 1


def test_pixel_inspection_coordinates_and_submatrix():
    """Verify pixel inspection returns correct localized coordinates, deltas, and submatrix window."""
    img_id = create_test_image(32, 32, (100, 150, 200))

    req = {
        "method": "additive",
        "base_source": "original",
        "channels": ["R", "G", "B"],
        "epsilon": 10.0,
        "additive_mode": "positive_const",
        "selected_point": [16, 16],
        "submatrix_radius": 2,
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200
    data = res.json()

    inspection = data["pixel_inspection"]
    assert inspection is not None
    assert inspection["x"] == 16
    assert inspection["y"] == 16
    assert inspection["original_rgb"] == [100, 150, 200]
    assert inspection["perturbed_rgb"] == [110, 160, 210]
    assert inspection["delta_rgb"] == [10.0, 10.0, 10.0]

    # Submatrix radius 2 means 5x5 window
    bounds = inspection["submatrix_bounds"]
    assert bounds["xmin"] == 14
    assert bounds["xmax"] == 18
    assert bounds["ymin"] == 14
    assert bounds["ymax"] == 18
    assert len(inspection["submatrix_delta_r"]) == 5
    assert len(inspection["submatrix_delta_r"][0]) == 5


def test_perturbed_tensor_export():
    """Verify exporting perturbed tensors as .npy, .npz, and .csv."""
    img_id = create_test_image(16, 16, (90, 120, 150))

    # Apply perturbation first
    res_pert = client.post(
        f"/api/images/{img_id}/perturb",
        json={"method": "gaussian", "sigma": 10.0, "seed": 42},
    )
    assert res_pert.status_code == 200

    # 1. Download perturbed_rgb .npy
    res_npy = client.get(f"/api/images/{img_id}/tensors/download?target=perturbed_rgb&format=npy")
    assert res_npy.status_code == 200
    arr_npy = np.load(io.BytesIO(res_npy.content))
    stored = image_service.get_image(img_id)
    assert np.array_equal(arr_npy, stored.perturbed_rgb_array)

    # 2. Download difference .npy
    res_diff = client.get(f"/api/images/{img_id}/tensors/download?target=difference&format=npy")
    assert res_diff.status_code == 200
    arr_diff = np.load(io.BytesIO(res_diff.content))
    assert np.allclose(arr_diff, stored.perturbation_delta_array)

    # 3. Download NPZ archive containing all
    res_npz = client.get(f"/api/images/{img_id}/tensors/download?target=all&format=npz")
    assert res_npz.status_code == 200
    npz_data = np.load(io.BytesIO(res_npz.content))
    assert "perturbed_rgb" in npz_data
    assert "perturbation" in npz_data
    assert "original_rgb" in npz_data


def test_perturbation_metadata_endpoint():
    """Verify perturbation metadata endpoint returns 404 before perturbation and 200 after."""
    img_id = create_test_image(16, 16, (100, 100, 100))

    # Before perturbation
    res_before = client.get(f"/api/images/{img_id}/perturbation/metadata")
    assert res_before.status_code == 404

    # Apply perturbation
    res_pert = client.post(
        f"/api/images/{img_id}/perturb",
        json={"method": "inversion"},
    )
    assert res_pert.status_code == 200

    # After perturbation
    res_after = client.get(f"/api/images/{img_id}/perturbation/metadata")
    assert res_after.status_code == 200
    meta = res_after.json()
    assert meta["method"] == "inversion"
    assert "metrics" in meta


def test_perturb_transformed_image_pipeline():
    """Verify pipeline perturbing a Phase 3 transformed image."""
    img_id = create_test_image(32, 32, (100, 150, 200))

    # Expect 400 when attempting to perturb transformed before applying any transformation
    res_fail = client.post(
        f"/api/images/{img_id}/perturb",
        json={"method": "additive", "base_source": "transformed"},
    )
    assert res_fail.status_code == 400

    # Apply Phase 3 transformation (rotation 90 deg)
    res_trans = client.post(
        f"/api/images/{img_id}/transform",
        json={
            "operations": [{"type": "rotate", "params": {"angle": 90.0}}],
            "interpolation": "nearest",
            "canvas_mode": "same",
        },
    )
    assert res_trans.status_code == 200

    # Now perturb the transformed image
    res_pert = client.post(
        f"/api/images/{img_id}/perturb",
        json={"method": "gaussian", "base_source": "transformed", "sigma": 10.0, "seed": 42},
    )
    assert res_pert.status_code == 200
    data = res_pert.json()
    assert data["base_source"] == "transformed"
    assert data["metrics"]["mse"] > 0


def test_brightness_and_contrast_operation():
    """Verify mathematically defined brightness & contrast adjustment."""
    # Test on known mid-gray image (128, 128, 128)
    img_id = create_test_image(16, 16, (128, 128, 128))

    # With c=1.5 and b=+20: 1.5 * (128 - 128) + 128 + 20 = 148
    req = {
        "operations": [
            {
                "type": "brightness_contrast",
                "enabled": True,
                "weight": 1.0,
                "channels": ["R", "G", "B"],
                "params": {"contrast_factor": 1.5, "brightness_offset": 20.0},
            }
        ]
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["intermediate_steps"][0]["category"] == "photometric"

    stored = image_service.get_image(img_id)
    assert np.all(stored.perturbed_rgb_array == 148)


def test_sequential_composition_pipeline_and_intermediates():
    """Verify sequential composition produces intermediate results and chains operations."""
    img_id = create_test_image(16, 16, (100, 100, 100))

    req = {
        "composition_mode": "sequential",
        "operations": [
            {
                "id": "op1",
                "type": "additive",
                "enabled": True,
                "weight": 1.0,
                "channels": ["R", "G", "B"],
                "params": {"epsilon": 20.0, "additive_mode": "positive_const"},
            },
            {
                "id": "op2",
                "type": "inversion",
                "enabled": True,
                "weight": 1.0,
                "channels": ["R", "G", "B"],
                "params": {},
            },
        ],
        "selected_point": [8, 8],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200
    data = res.json()

    # Step 1: 100 + 20 = 120
    # Step 2: 255 - 120 = 135
    stored = image_service.get_image(img_id)
    assert np.all(stored.perturbed_rgb_array == 135)
    assert len(data["intermediate_steps"]) == 2
    assert data["intermediate_steps"][0]["step_index"] == 1
    assert data["intermediate_steps"][1]["step_index"] == 2

    # Check pixel inspection step tracking
    pi = data["pixel_inspection"]
    assert pi is not None
    assert len(pi["intermediate_pixels"]) == 2
    assert pi["intermediate_pixels"][0]["rgb"] == [120, 120, 120]
    assert pi["intermediate_pixels"][1]["rgb"] == [135, 135, 135]


def test_sequential_composition_non_commutative():
    """Verify that composition order matters due to clipping and operation sequence (AB != BA)."""
    img_id = create_test_image(16, 16, (100, 100, 100))

    # Order A: Additive (+50) then Inversion
    # Step 1: 100 + 50 = 150
    # Step 2: 255 - 150 = 105
    req_a = {
        "composition_mode": "sequential",
        "operations": [
            {"type": "additive", "params": {"epsilon": 50.0, "additive_mode": "positive_const"}},
            {"type": "inversion", "params": {}},
        ],
    }
    res_a = client.post(f"/api/images/{img_id}/perturb", json=req_a)
    arr_a = image_service.get_image(img_id).perturbed_rgb_array.copy()

    # Order B: Inversion then Additive (+50)
    # Step 1: 255 - 100 = 155
    # Step 2: 155 + 50 = 205
    req_b = {
        "composition_mode": "sequential",
        "operations": [
            {"type": "inversion", "params": {}},
            {"type": "additive", "params": {"epsilon": 50.0, "additive_mode": "positive_const"}},
        ],
    }
    res_b = client.post(f"/api/images/{img_id}/perturb", json=req_b)
    arr_b = image_service.get_image(img_id).perturbed_rgb_array.copy()

    assert not np.array_equal(arr_a, arr_b), "Sequential composition must be non-commutative (order matters)"
    assert np.all(arr_a == 105)
    assert np.all(arr_b == 205)


def test_weighted_perturbation_combination():
    """Verify weighted mode computes deltas relative to base image and linearly superimposes them."""
    img_id = create_test_image(16, 16, (100, 100, 100))

    # Base image = 100.
    # Op 1: Additive (+10), weight = 2.0 -> delta_1 = +10, weighted = +20
    # Op 2: Additive (-5), weight = 3.0 -> delta_2 = -5, weighted = -15
    # Combined delta = +20 + (-15) = +5
    # Result = 100 + 5 = 105
    req = {
        "composition_mode": "weighted",
        "operations": [
            {
                "type": "additive",
                "weight": 2.0,
                "params": {"epsilon": 10.0, "additive_mode": "positive_const"},
            },
            {
                "type": "additive",
                "weight": 3.0,
                "params": {"epsilon": 5.0, "additive_mode": "negative_const"},
            },
        ],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200

    stored = image_service.get_image(img_id)
    assert np.all(stored.perturbed_rgb_array == 105)


def test_weighted_global_perturbation_bound():
    """Verify that global bound ||Δ_combined||_∞ <= ε_global clamps weighted deltas."""
    img_id = create_test_image(16, 16, (100, 100, 100))

    # Weight 5.0 on +20 shift -> +100 shift, but global bound = 15.0
    req = {
        "composition_mode": "weighted",
        "global_bound": 15.0,
        "operations": [
            {
                "type": "additive",
                "weight": 5.0,
                "params": {"epsilon": 20.0, "additive_mode": "positive_const"},
            }
        ],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200
    data = res.json()

    assert data["metrics"]["linf_norm"] <= 15.0 + 1e-5
    stored = image_service.get_image(img_id)
    # 100 + 15 = 115
    assert np.all(stored.perturbed_rgb_array == 115)


def test_disabled_operations_in_pipeline():
    """Verify disabled operations produce zero contribution."""
    img_id = create_test_image(16, 16, (100, 100, 100))

    req = {
        "composition_mode": "sequential",
        "operations": [
            {
                "type": "additive",
                "enabled": False,
                "params": {"epsilon": 50.0, "additive_mode": "positive_const"},
            },
            {
                "type": "additive",
                "enabled": True,
                "params": {"epsilon": 10.0, "additive_mode": "positive_const"},
            },
        ],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200

    stored = image_service.get_image(img_id)
    # Only enabled step (+10) applies: 100 + 10 = 110
    assert np.all(stored.perturbed_rgb_array == 110)


def test_intermediate_tensors_and_metadata_export():
    """Verify intermediate tensors and metadata are exportable via download endpoint."""
    img_id = create_test_image(16, 16, (90, 90, 90))

    req = {
        "composition_mode": "sequential",
        "operations": [
            {"type": "additive", "params": {"epsilon": 10.0, "additive_mode": "positive_const"}},
            {"type": "additive", "params": {"epsilon": 10.0, "additive_mode": "positive_const"}},
        ],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200

    # 1. Download intermediate_step_1
    res_step1 = client.get(f"/api/images/{img_id}/tensors/download?target=intermediate_step_1&format=npy")
    assert res_step1.status_code == 200
    arr_step1 = np.load(io.BytesIO(res_step1.content))
    assert np.all(arr_step1 == 100)

    # 2. Download all in NPZ
    res_npz = client.get(f"/api/images/{img_id}/tensors/download?target=all&format=npz")
    assert res_npz.status_code == 200
    npz_data = np.load(io.BytesIO(res_npz.content))
    assert "intermediate_step_1" in npz_data
    assert "intermediate_step_2" in npz_data
    assert "perturbed_rgb" in npz_data

    # 3. Download JSON metadata
    res_json = client.get(f"/api/images/{img_id}/tensors/download?target=metadata&format=json")
    assert res_json.status_code == 200
    meta_json = res_json.json()
    assert meta_json["image_shape"] == [16, 16, 3]
    assert "binary_format_notice" in meta_json
    assert meta_json["pipeline_metadata"]["composition_mode"] == "sequential"



def test_sequential_cumulative_global_budget_enforcement():
    """Verify sequential mode enforces cumulative budget relative to base image at every stage."""
    img_id = create_test_image(16, 16, (100, 100, 100))

    # Two consecutive +8.0 shifts would sum to +16.0, but budget bounds cumulative delta to 11.0
    req = {
        "composition_mode": "sequential",
        "budget": {
            "type": "linf",
            "epsilon_linf": 11.0,
            "unit": "pixel_0_255",
        },
        "operations": [
            {
                "type": "additive",
                "params": {"epsilon": 8.0, "additive_mode": "positive_const"},
            },
            {
                "type": "additive",
                "params": {"epsilon": 8.0, "additive_mode": "positive_const"},
            },
        ],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200
    data = res.json()

    assert data["metrics"]["linf_norm"] <= 11.0 + 1e-5
    stored = image_service.get_image(img_id)
    # Stage 1: 100 + 8 = 108; Stage 2: 100 + min(16, 11) = 111
    assert np.all(stored.perturbed_rgb_array == 111)


def test_l2_ball_projection_across_tensor():
    """Verify L2 ball projection scales delta tensor down to radius epsilon_l2."""
    img_id = create_test_image(10, 10, (120, 120, 120))
    # 10x10x3 = 300 elements. Shift of +10 gives raw L2 = sqrt(300 * 100) ≈ 173.2
    target_l2 = 25.0

    req = {
        "composition_mode": "weighted",
        "budget": {
            "type": "l2",
            "epsilon_l2": target_l2,
            "unit": "pixel_0_255",
        },
        "operations": [
            {
                "type": "additive",
                "weight": 1.0,
                "params": {"epsilon": 10.0, "additive_mode": "positive_const"},
            }
        ],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200
    data = res.json()

    assert data["budget_report"] is not None
    assert data["budget_report"]["projected"] is True
    assert data["budget_report"]["post_projection_l2"] <= target_l2 + 1e-4


def test_budget_unit_normalized_vs_pixel():
    """Verify normalized [0, 1] budget converts accurately without naive auto-division."""
    img_id = create_test_image(16, 16, (100, 100, 100))
    # eps = 16/255 in normalized [0, 1] equals 16.0 pixel units
    eps_normalized = 16.0 / 255.0

    req = {
        "composition_mode": "sequential",
        "budget": {
            "type": "linf",
            "epsilon_linf": eps_normalized,
            "unit": "normalized_0_1",
        },
        "operations": [
            {
                "type": "additive",
                "params": {"epsilon": 50.0, "additive_mode": "positive_const"},
            }
        ],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200
    data = res.json()

    assert data["metrics"]["linf_norm"] <= 16.0 + 1e-4
    stored = image_service.get_image(img_id)
    assert np.all(stored.perturbed_rgb_array == 116)


def test_l0_threshold_and_spatial_vs_element_detection():
    """Verify L0 measurement distinguishes scalar elements from spatial pixels with threshold."""
    img_id = create_test_image(16, 16, (100, 100, 100))
    # Perturb channel R only by +5.0
    req = {
        "composition_mode": "sequential",
        "l0_threshold": 2.0,
        "operations": [
            {
                "type": "additive",
                "channels": ["R"],
                "params": {"epsilon": 5.0, "additive_mode": "positive_const"},
            }
        ],
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200
    metrics = res.json()["metrics"]

    # Total pixels = 256; Total elements = 768 (256 * 3)
    assert metrics["total_pixels_count"] == 256
    assert metrics["total_elements_count"] == 768
    # Only R channel changed (> 2.0), so 256 elements changed
    assert metrics["changed_elements_count"] == 256
    # All 256 spatial pixels have at least one channel changed
    assert metrics["changed_pixels_count"] == 256

    # Test with threshold higher than delta: threshold 10.0 > 5.0
    req_high_thresh = {
        "composition_mode": "sequential",
        "l0_threshold": 10.0,
        "operations": [
            {
                "type": "additive",
                "channels": ["R"],
                "params": {"epsilon": 5.0, "additive_mode": "positive_const"},
            }
        ],
    }
    res2 = client.post(f"/api/images/{img_id}/perturb", json=req_high_thresh)
    assert res2.status_code == 200
    metrics2 = res2.json()["metrics"]
    assert metrics2["changed_elements_count"] == 0
    assert metrics2["changed_pixels_count"] == 0


def test_quantization_distortion_and_channel_metrics():
    """Verify quantization distortion metrics and per-channel statistics breakdown."""
    img_id = create_test_image(16, 16, (100, 120, 140))
    req = {
        "method": "additive",
        "epsilon": 12.0,
        "additive_mode": "random",
        "seed": 42,
    }
    res = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res.status_code == 200
    metrics = res.json()["metrics"]

    # Quantization distortion
    assert metrics["quantization_distortion"] is not None
    assert "mae" in metrics["quantization_distortion"]
    assert "mse" in metrics["quantization_distortion"]
    assert "linf" in metrics["quantization_distortion"]
    assert "std" in metrics["quantization_distortion"]

    # Per-channel metrics
    assert metrics["per_channel"] is not None
    assert "R" in metrics["per_channel"]
    assert "G" in metrics["per_channel"]
    assert "B" in metrics["per_channel"]
    assert metrics["per_channel"]["R"]["linf_norm"] <= 12.0 + 1e-5


def test_per_step_individual_random_seed():
    """Verify individual per-step seeds produce deterministic step outputs."""
    img_id = create_test_image(16, 16, (100, 100, 100))

    req = {
        "composition_mode": "sequential",
        "seed": 999,  # pipeline seed
        "operations": [
            {
                "type": "gaussian",
                "seed": 4321,  # step 1 seed
                "params": {"sigma": 10.0},
            },
            {
                "type": "gaussian",
                "seed": 8765,  # step 2 seed
                "params": {"sigma": 15.0},
            },
        ],
    }
    res1 = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res1.status_code == 200

    # Execute again with different pipeline seed, but identical step seeds
    req["seed"] = 111
    res2 = client.post(f"/api/images/{img_id}/perturb", json=req)
    assert res2.status_code == 200

    assert res1.json()["perturbed_rgb_data_url"] == res2.json()["perturbed_rgb_data_url"]


def test_phase3_readiness_and_bidirectional_composition():
    """Verify Phase 3 readiness steps and Perturb -> Transform bidirectional composition."""
    img_id = create_test_image(32, 32, (100, 100, 100))

    # 1. Apply perturbation first
    pert_req = {
        "method": "additive",
        "epsilon": 20.0,
        "additive_mode": "positive_const",
    }
    res_pert = client.post(f"/api/images/{img_id}/perturb", json=pert_req)
    assert res_pert.status_code == 200

    # 2. Apply transformation on base_source='perturbed'
    trans_req = {
        "base_source": "perturbed",
        "operations": [
            {
                "type": "random_affine",
                "params": {"seed": 777, "max_angle": 15.0, "max_scale": 1.1},
            },
            {
                "type": "resize",
                "params": {"scale_factor": 0.5},
            },
            {
                "type": "jpeg_compression",
                "params": {"quality": 75},
            },
        ],
    }
    res_trans = client.post(f"/api/images/{img_id}/transform", json=trans_req)
    assert res_trans.status_code == 200
    trans_data = res_trans.json()

    assert len(trans_data["steps"]) == 3
    assert trans_data["transformed_width"] > 0
    assert trans_data["transformed_height"] > 0
    stored = image_service.get_image(img_id)
    assert stored.transformed_rgb_array is not None
    assert stored.transformed_rgb_array.shape == (trans_data["transformed_height"], trans_data["transformed_width"], 3)
