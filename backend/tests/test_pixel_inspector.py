"""Regression tests for RGB Matrix Representation and Pixel Inspector submatrix neighborhood."""

import io
import pytest
import numpy as np
from PIL import Image
from app.services.image_service import image_service
from app.services.matrix_service import matrix_service
from app.api.routes import load_sample_image, inspect_pixel
from app.models.schemas import PixelInspectionRequest


def test_known_3x3_distinct_rgb_values():
    """
    Task 6 & 7: Verify pixel inspector on a known 3x3 image containing 9 completely distinct RGB values.
    Confirms neighborhood output exactly matches the expected matrix and does NOT reuse the center pixel.
    """
    # 3x3 image array: shape (3, 3, 3)
    # Row 0 (y=0): x=0: [10,20,30],    x=1: [40,50,60],    x=2: [70,80,90]
    # Row 1 (y=1): x=0: [100,110,120], x=1: [130,140,150], x=2: [160,170,180]
    # Row 2 (y=2): x=0: [190,200,210], x=1: [220,230,240], x=2: [250,255,15]
    arr_3x3 = np.array([
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        [[190, 200, 210], [220, 230, 240], [250, 255, 15]]
    ], dtype=np.uint8)

    pil_img = Image.fromarray(arr_3x3, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    meta = image_service.upload_image("test_distinct_3x3.png", buf.getvalue())
    stored = image_service.get_image(meta.image_id)

    # Inspect center pixel at x=1, y=1 with radius=1 (yielding full 3x3 window)
    res = matrix_service.inspect_pixel(stored, x=1, y=1, radius=1)

    # Center pixel verification
    assert res.x == 1
    assert res.y == 1
    assert res.r == 130
    assert res.g == 140
    assert res.b == 150

    # Grayscale calculation verification: Y = 0.299*130 + 0.587*140 + 0.114*150 = 38.87 + 82.18 + 17.10 = 138.15
    assert np.isclose(res.grayscale_exact, 138.15, atol=1e-2)
    assert res.grayscale_uint8 == 138

    # Verify R channel submatrix
    expected_r = [[10, 40, 70], [100, 130, 160], [190, 220, 250]]
    assert res.submatrix_r == expected_r, f"Expected {expected_r}, got {res.submatrix_r}"

    # Verify G channel submatrix
    expected_g = [[20, 50, 80], [110, 140, 170], [200, 230, 255]]
    assert res.submatrix_g == expected_g, f"Expected {expected_g}, got {res.submatrix_g}"

    # Verify B channel submatrix
    expected_b = [[30, 60, 90], [120, 150, 180], [210, 240, 15]]
    assert res.submatrix_b == expected_b, f"Expected {expected_b}, got {res.submatrix_b}"

    # Verify Grayscale submatrix matches Y = 0.299R + 0.587G + 0.114B for all 9 cells
    expected_gray = np.round(
        0.299 * arr_3x3[:, :, 0] + 0.587 * arr_3x3[:, :, 1] + 0.114 * arr_3x3[:, :, 2], 2
    ).tolist()
    assert res.submatrix_grayscale == expected_gray

    # Verify all 9 neighborhood entries are distinct (no repeated center pixel)
    flattened_rgb = [
        (res.submatrix_r[r][c], res.submatrix_g[r][c], res.submatrix_b[r][c])
        for r in range(3) for c in range(3)
    ]
    assert len(set(flattened_rgb)) == 9, "All 9 cells must be distinct!"


def test_rectangular_asymmetric_image_indexing():
    """
    Task 3: Check image array indexing convention image[y, x], not image[x, y].
    Uses a non-square image (H=4, W=7) to ensure no row/column axis transpositions occur.
    """
    h, w = 4, 7
    # Unique values: R = y*10 + x
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    r_chan = (y_coords * 10 + x_coords).astype(np.uint8)
    g_chan = (100 + y_coords * 10 + x_coords).astype(np.uint8)
    b_chan = (200 - (y_coords * 10 + x_coords)).astype(np.uint8)

    arr_rect = np.stack([r_chan, g_chan, b_chan], axis=2)
    pil_img = Image.fromarray(arr_rect, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    meta = image_service.upload_image("rect_4x7.png", buf.getvalue())
    stored = image_service.get_image(meta.image_id)

    # Inspect coordinate x=5, y=2 (note x > h!)
    res = matrix_service.inspect_pixel(stored, x=5, y=2, radius=1)

    assert res.x == 5
    assert res.y == 2
    assert res.r == 2 * 10 + 5  # 25
    assert res.g == 125
    assert res.b == 175

    # Bounds: row 1..3 (size 3), col 4..6 (size 3)
    assert res.submatrix_bounds == {
        "row_start": 1,
        "row_end": 3,
        "col_start": 4,
        "col_end": 6,
    }

    # Verify R slice: rows 1..3, cols 4..6
    expected_r_slice = r_chan[1:4, 4:7].tolist()
    assert res.submatrix_r == expected_r_slice


def test_window_sizes_3x3_5x5_7x7_9x9():
    """
    Task 9: Verify 3x3 (radius=1), 5x5 (radius=2), 7x7 (radius=3), and 9x9 (radius=4) windows.
    """
    h, w = 30, 30
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    # Unique spatial gradient
    r_chan = (x_coords * 8).astype(np.uint8)
    g_chan = (y_coords * 8).astype(np.uint8)
    b_chan = ((x_coords + y_coords) * 4).astype(np.uint8)

    arr = np.stack([r_chan, g_chan, b_chan], axis=2)
    pil_img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    meta = image_service.upload_image("gradient_30x30.png", buf.getvalue())
    stored = image_service.get_image(meta.image_id)

    cx, cy = 15, 15
    for r in [1, 2, 3, 4]:
        dim = 2 * r + 1
        res = matrix_service.inspect_pixel(stored, x=cx, y=cy, radius=r)
        assert len(res.submatrix_r) == dim
        assert len(res.submatrix_r[0]) == dim
        assert len(res.submatrix_g) == dim
        assert len(res.submatrix_b) == dim
        assert len(res.submatrix_grayscale) == dim

        # Verify values match exact array slices
        assert np.array_equal(res.submatrix_r, r_chan[cy-r:cy+r+1, cx-r:cx+r+1])
        assert np.array_equal(res.submatrix_g, g_chan[cy-r:cy+r+1, cx-r:cx+r+1])
        assert np.array_equal(res.submatrix_b, b_chan[cy-r:cy+r+1, cx-r:cx+r+1])


def test_boundary_coordinates():
    """
    Task 10: Check boundary coordinates such as x=0, y=0 and x=W-1, y=H-1.
    """
    h, w = 16, 16
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[0, 0] = [255, 1, 1]        # Top-left
    arr[0, w - 1] = [1, 255, 1]    # Top-right
    arr[h - 1, 0] = [1, 1, 255]    # Bottom-left
    arr[h - 1, w - 1] = [255, 255, 255]  # Bottom-right

    pil_img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    meta = image_service.upload_image("boundaries_16x16.png", buf.getvalue())
    stored = image_service.get_image(meta.image_id)

    # 1. Top-Left (0, 0) with radius=3 -> yields 4x4 submatrix (rows 0..3, cols 0..3)
    res_tl = matrix_service.inspect_pixel(stored, x=0, y=0, radius=3)
    assert res_tl.submatrix_bounds == {"row_start": 0, "row_end": 3, "col_start": 0, "col_end": 3}
    assert len(res_tl.submatrix_r) == 4
    assert len(res_tl.submatrix_r[0]) == 4
    assert res_tl.r == 255 and res_tl.g == 1 and res_tl.b == 1

    # 2. Bottom-Right (15, 15) with radius=3 -> yields 4x4 submatrix (rows 12..15, cols 12..15)
    res_br = matrix_service.inspect_pixel(stored, x=15, y=15, radius=3)
    assert res_br.submatrix_bounds == {"row_start": 12, "row_end": 15, "col_start": 12, "col_end": 15}
    assert len(res_br.submatrix_r) == 4
    assert len(res_br.submatrix_r[0]) == 4
    assert res_br.r == 255 and res_br.g == 255 and res_br.b == 255


def test_harmonic_benchmark_center_disk_ground_truth():
    """
    Task 1, 2, 4, 5: Verify the actual sample image data.
    Confirms that at (128, 128) in the harmonic benchmark image, every cell in the 7x7
    neighborhood has [230, 180, 50] because the benchmark contains a solid disk of radius 50.
    Also confirms that outside this disk (e.g. at (50, 50)), neighborhood cells vary.
    """
    meta = load_sample_image(preset="harmonic")

    # Inspect center (128, 128) with 7x7 window (radius=3)
    req_center = PixelInspectionRequest(x=128, y=128, submatrix_radius=3)
    res_center = inspect_pixel(meta.image_id, req_center)

    # 1. Confirm center pixel values
    assert res_center.r == 230
    assert res_center.g == 180
    assert res_center.b == 50

    # 2. Confirm all cells in 7x7 window around (128, 128) are legitimately [230, 180, 50]
    # because max distance from (128, 128) in a 7x7 window is sqrt(3^2 + 3^2) ~ 4.24 < 50
    for r_idx in range(7):
        for c_idx in range(7):
            assert res_center.submatrix_r[r_idx][c_idx] == 230
            assert res_center.submatrix_g[r_idx][c_idx] == 180
            assert res_center.submatrix_b[r_idx][c_idx] == 50

    # 3. Inspect outside the disk at (50, 50) where distance from center is sqrt(78^2 + 78^2) ~ 110.3 > 50
    req_outside = PixelInspectionRequest(x=50, y=50, submatrix_radius=3)
    res_outside = inspect_pixel(meta.image_id, req_outside)

    # Confirm that outside the disk, neighborhood cells have varying values (not repeated center pixel)
    distinct_r_values = set(res_outside.submatrix_r[0])
    assert len(distinct_r_values) > 1, "Neighborhood cells outside disk must have distinct harmonic wave values!"

    # 4. Inspect disk boundary at (128, 177) where y=177 (dist=49 < 50) transitions to y=178 (dist=50 >= 50)
    req_boundary = PixelInspectionRequest(x=128, y=177, submatrix_radius=3)
    res_boundary = inspect_pixel(meta.image_id, req_boundary)

    # y=174..177 are inside disk (value 230), y=178..180 are outside disk (harmonic wave values != 230)
    row_inside = res_boundary.submatrix_r[3]   # y=177 (index 3)
    row_outside = res_boundary.submatrix_r[4]  # y=178 (index 4)
    assert row_inside == [230, 230, 230, 230, 230, 230, 230]
    assert any(val != 230 for val in row_outside), "Values across disk boundary must change from 230"

