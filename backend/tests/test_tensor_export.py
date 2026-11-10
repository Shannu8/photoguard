"""Unit and integration tests for Phase 3: Tensor Export System."""

import io
import zipfile
import pytest
import numpy as np
from app.services.image_service import image_service
from app.api.routes import load_sample_image, transform_image, download_tensor, get_tensor_metadata
from app.models.schemas import TransformationStep, TransformationRequest


def test_npy_tensor_export_and_direct_loading():
    """Verify downloading original and transformed RGB tensors as .npy files loadable with np.load()."""
    meta = load_sample_image()
    stored = image_service.get_image(meta.image_id)

    # Apply a rotation so transformed tensor exists
    req = TransformationRequest(
        operations=[TransformationStep(type="rotate", params={"angle": 90.0})],
        canvas_mode="same"
    )
    transform_image(meta.image_id, req)

    # 1. Export Original RGB (.npy)
    res_orig = download_tensor(meta.image_id, target="original_rgb", format="npy")
    assert res_orig.media_type == "application/octet-stream"
    assert "attachment" in res_orig.headers["Content-Disposition"]
    assert ".npy" in res_orig.headers["Content-Disposition"]

    bio = io.BytesIO(res_orig.body)
    loaded_orig = np.load(bio)

    # Verify exact tensor integrity
    assert loaded_orig.shape == (256, 256, 3), f"Expected (256, 256, 3), got {loaded_orig.shape}"
    assert loaded_orig.dtype == np.uint8, f"Expected uint8, got {loaded_orig.dtype}"
    assert np.array_equal(loaded_orig, stored.rgb_array), "Exported array must equal in-memory source array"

    # 2. Export Transformed RGB (.npy)
    res_trans = download_tensor(meta.image_id, target="transformed_rgb", format="npy")
    bio_trans = io.BytesIO(res_trans.body)
    loaded_trans = np.load(bio_trans)
    assert loaded_trans.shape == (256, 256, 3)
    assert np.array_equal(loaded_trans, stored.transformed_rgb_array)

    # 3. Export Original Grayscale (.npy)
    res_gray = download_tensor(meta.image_id, target="original_grayscale", format="npy")
    bio_gray = io.BytesIO(res_gray.body)
    loaded_gray = np.load(bio_gray)
    assert loaded_gray.shape == (256, 256)
    assert loaded_gray.dtype == np.float64
    assert np.array_equal(loaded_gray, stored.grayscale_array)


def test_npz_archive_export():
    """Verify downloading all tensors in a single .npz multi-array archive."""
    meta = load_sample_image()
    req = TransformationRequest(
        operations=[TransformationStep(type="scale", params={"sx": 1.5, "sy": 1.5})],
        canvas_mode="fit"
    )
    transform_image(meta.image_id, req)

    res = download_tensor(meta.image_id, target="all", format="npz")
    assert res.media_type == "application/octet-stream"
    assert ".npz" in res.headers["Content-Disposition"]

    bio = io.BytesIO(res.body)
    npz_data = np.load(bio)

    assert "original_rgb" in npz_data
    assert "original_grayscale" in npz_data
    assert "transformed_rgb" in npz_data
    assert "transformed_grayscale" in npz_data

    stored = image_service.get_image(meta.image_id)
    assert np.array_equal(npz_data["original_rgb"], stored.rgb_array)
    assert np.array_equal(npz_data["original_grayscale"], stored.grayscale_array)
    assert np.array_equal(npz_data["transformed_rgb"], stored.transformed_rgb_array)
    assert np.array_equal(npz_data["transformed_grayscale"], stored.transformed_grayscale_array)


def test_csv_matrix_and_channel_export():
    """Verify CSV export for 2D grayscale and multi-channel RGB zip archive."""
    meta = load_sample_image()

    # 1. 2D Grayscale CSV
    res_gray_csv = download_tensor(meta.image_id, target="original_grayscale", format="csv")
    assert res_gray_csv.media_type == "text/csv"
    assert ".csv" in res_gray_csv.headers["Content-Disposition"]

    sio = io.StringIO(res_gray_csv.body.decode("utf-8"))
    loaded_csv_matrix = np.loadtxt(sio, delimiter=",")
    stored = image_service.get_image(meta.image_id)
    assert loaded_csv_matrix.shape == (256, 256)
    assert np.allclose(loaded_csv_matrix, stored.grayscale_array, atol=1e-3)

    # 2. 3D RGB CSV (packaged into zip archive with red, green, blue channel CSVs)
    res_rgb_csv = download_tensor(meta.image_id, target="original_rgb", format="csv")
    assert res_rgb_csv.media_type == "application/zip"
    assert ".zip" in res_rgb_csv.headers["Content-Disposition"]

    bio_zip = io.BytesIO(res_rgb_csv.body)
    with zipfile.ZipFile(bio_zip, "r") as zf:
        file_list = zf.namelist()
        assert any("channel_red.csv" in f for f in file_list)
        assert any("channel_green.csv" in f for f in file_list)
        assert any("channel_blue.csv" in f for f in file_list)

        r_csv = zf.read([f for f in file_list if "channel_red.csv" in f][0]).decode("utf-8")
        loaded_r = np.loadtxt(io.StringIO(r_csv), delimiter=",")
        assert np.array_equal(loaded_r, stored.rgb_array[:, :, 0])


def test_tensor_metadata_endpoint():
    """Verify tensor metadata endpoint accurately reports shapes, element counts, and data types."""
    meta = load_sample_image()
    meta_before = get_tensor_metadata(meta.image_id)
    assert meta_before.has_transformed is False
    assert meta_before.original_rgb.shape == [256, 256, 3]
    assert meta_before.original_rgb.total_elements == 256 * 256 * 3
    assert meta_before.original_grayscale.shape == [256, 256]

    # After transformation
    req = TransformationRequest(
        operations=[TransformationStep(type="reflect", params={"axis": "horizontal"})],
        canvas_mode="same"
    )
    transform_image(meta.image_id, req)

    meta_after = get_tensor_metadata(meta.image_id)
    assert meta_after.has_transformed is True
    assert meta_after.transformed_rgb is not None
    assert meta_after.transformed_rgb.shape == [256, 256, 3]
    assert meta_after.transformed_grayscale is not None
    assert meta_after.transformed_grayscale.shape == [256, 256]

