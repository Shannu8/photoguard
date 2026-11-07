"""Integration tests directly validating API route handler logic and responses."""

import pytest
from app.api.routes import (
    load_sample_image,
    get_image_metadata,
    inspect_pixel,
    get_image_statistics,
    get_matrix_rank,
    get_eigen_analysis,
    get_svd_analysis,
    reconstruct_image,
)
from app.models.schemas import (
    PixelInspectionRequest,
    EigenAnalysisRequest,
    ReconstructionRequest,
)


def test_api_handlers_pipeline():
    # 1. Load sample image
    meta = load_sample_image()
    image_id = meta.image_id
    assert meta.width == 256
    assert meta.height == 256
    assert meta.channels == 3
    assert meta.total_pixels == 256 * 256

    # 2. Retrieve metadata
    meta_retrieved = get_image_metadata(image_id)
    assert meta_retrieved.image_id == image_id

    # 3. Inspect Pixel
    req_inspect = PixelInspectionRequest(x=50, y=50, submatrix_radius=2)
    inspect_data = inspect_pixel(image_id, req_inspect)
    assert inspect_data.x == 50
    assert inspect_data.y == 50
    assert "Y = 0.299" in inspect_data.calculation_steps
    assert len(inspect_data.submatrix_r) == 5

    # 4. Statistics
    stats = get_image_statistics(image_id)
    assert len(stats.grayscale.histogram) == 256
    assert len(stats.red.histogram) == 256

    # 5. Matrix Rank
    rank_data = get_matrix_rank(image_id)
    assert rank_data.rows == 256
    assert rank_data.cols == 256
    assert rank_data.rank > 0
    assert "CRITICAL MATHEMATICAL DISTINCTION" in rank_data.scientific_notes

    # 6. Eigen Analysis (Gram Row)
    req_eigen = EigenAnalysisRequest(matrix_type="gram_row", max_components=20)
    eigen_data = get_eigen_analysis(image_id, req_eigen)
    assert len(eigen_data.eigenvalues_real) > 0
    assert "MATHEMATICAL DISTINCTION" in eigen_data.pca_distinction

    # 7. SVD Analysis
    svd_data = get_svd_analysis(image_id)
    assert len(svd_data.singular_values) == 256
    assert svd_data.effective_rank > 0

    # 8. Low-Rank Reconstruction (RGB)
    req_recon = ReconstructionRequest(k=20, mode="rgb")
    recon_data = reconstruct_image(image_id, req_recon)
    assert recon_data.k == 20
    assert recon_data.retained_percentage == round(20 / 256 * 100, 2)
    assert recon_data.psnr > 0
    assert "data:image/png;base64," in recon_data.reconstructed_data_url
    assert "data:image/png;base64," in recon_data.difference_data_url
