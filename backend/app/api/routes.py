"""REST API endpoints for mathematical image analysis."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path
from typing import Optional
import io
import numpy as np
from PIL import Image

from app.services.image_service import image_service
from app.services.matrix_service import matrix_service
from app.services.eigen_service import eigen_service
from app.services.svd_service import svd_service
from app.services.reconstruction_service import reconstruction_service
from app.models.schemas import (
    ImageMetadataResponse,
    PixelInspectionRequest,
    PixelInspectionResponse,
    ImageStatisticsResponse,
    MatrixRankResponse,
    EigenAnalysisRequest,
    EigenAnalysisResponse,
    SVDAnalysisResponse,
    ReconstructionRequest,
    ReconstructionResponse,
)

router = APIRouter(prefix="/api", tags=["Mathematical Image Analysis"])


@router.post("/upload", response_model=ImageMetadataResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file (PNG, JPEG, WebP, BMP).
    Validates format, preserves original bytes without alteration,
    and initializes mathematical representations.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Size limit guard (e.g. 20MB)
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (exceeds 20MB limit).")

    try:
        metadata = image_service.upload_image(file.filename, contents)
        return metadata
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


@router.get("/sample-image", response_model=ImageMetadataResponse)
def load_sample_image(preset: Optional[str] = Query(default="harmonic", description="Preset: 'harmonic', 'low_rank_blocks', or 'texture_noise'")):
    """
    Generates mathematically rich synthetic test images:
    - 'harmonic': Continuous 2D Fourier waves with geometric center (smooth SVD decay).
    - 'low_rank_blocks': Strictly low-rank step matrix (rank=3) showing exact rank deficiency and 0-error at k=3.
    - 'texture_noise': Multi-frequency Gaussian texture (slow SVD decay, high entropy).
    """
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]

    if preset == "low_rank_blocks":
        # Create an image from outer products of 3 orthogonal step vectors
        # Strictly rank 3!
        v1 = ((x // 64) % 2).astype(np.float64)
        v2 = ((x // 32) % 2).astype(np.float64)
        v3 = np.ones_like(v1)

        u1 = ((y // 64) % 2).astype(np.float64)
        u2 = ((y // 32) % 2).astype(np.float64)
        u3 = np.ones_like(u1)

        pattern = 70.0 * (u1 * v1) + 80.0 * (u2 * v2) + 50.0 * (u3 * v3)
        r_channel = np.clip(pattern + 30.0 * u1, 0, 255).astype(np.uint8)
        g_channel = np.clip(pattern + 20.0 * v2, 0, 255).astype(np.uint8)
        b_channel = np.clip(pattern, 0, 255).astype(np.uint8)
        filename = "low_rank_blocks_rank3_256x256.png"

    elif preset == "texture_noise":
        # High-entropy synthetic texture: superposition of varied frequencies
        r_f = 60.0 * np.sin(x / 4.0) + 60.0 * np.cos(y / 4.0) + 128.0
        g_f = 50.0 * np.sin(x / 2.0 + y / 2.0) + 50.0 * np.cos(x / 3.0) + 128.0
        b_f = 40.0 * np.sin(np.sqrt(x**2 + y**2) / 3.0) + 128.0
        r_channel = np.clip(r_f, 0, 255).astype(np.uint8)
        g_channel = np.clip(g_f, 0, 255).astype(np.uint8)
        b_channel = np.clip(b_f, 0, 255).astype(np.uint8)
        filename = "high_frequency_texture_256x256.png"

    else:  # "harmonic" default
        r_channel = (127.5 * (1 + np.sin(x / 16.0))).astype(np.uint8)
        g_channel = (127.5 * (1 + np.cos(y / 16.0))).astype(np.uint8)
        b_channel = (127.5 * (1 + np.sin((x + y) / 20.0))).astype(np.uint8)

        # Add circular disk in the center
        center_y, center_x = h // 2, w // 2
        dist_from_center = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        disk_mask = dist_from_center < 50
        r_channel[disk_mask] = 230
        g_channel[disk_mask] = 180
        b_channel[disk_mask] = 50
        filename = "harmonic_wave_benchmark_256x256.png"

    sample_rgb = np.stack([r_channel, g_channel, b_channel], axis=2)
    pil_img = Image.fromarray(sample_rgb, mode="RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    return image_service.upload_image(filename, raw_bytes)


@router.get("/images/{image_id}", response_model=ImageMetadataResponse)
def get_image_metadata(image_id: str = Path(...)):
    """Retrieve metadata for an existing image session."""
    try:
        return image_service.get_metadata(image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found.")


@router.post("/images/{image_id}/inspect-pixel", response_model=PixelInspectionResponse)
def inspect_pixel(
    image_id: str = Path(...),
    req: PixelInspectionRequest = ...,
):
    """
    Inspect a specific pixel at (x, y).
    Returns exact RGB values, transparent grayscale calculation, and local submatrix slice.
    """
    try:
        stored = image_service.get_image(image_id)
        return matrix_service.inspect_pixel(stored, req.x, req.y, req.submatrix_radius)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images/{image_id}/statistics", response_model=ImageStatisticsResponse)
def get_image_statistics(image_id: str = Path(...)):
    """Computes descriptive statistics and 256-bin histograms for all image channels."""
    try:
        stored = image_service.get_image(image_id)
        return matrix_service.get_image_statistics(stored)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images/{image_id}/matrix-rank", response_model=MatrixRankResponse)
def get_matrix_rank(image_id: str = Path(...)):
    """
    Calculates numerical rank of the grayscale image matrix,
    explains linear independence, and notes tolerance criteria.
    """
    try:
        stored = image_service.get_image(image_id)
        return matrix_service.analyze_matrix_rank(stored)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/{image_id}/eigen-analysis", response_model=EigenAnalysisResponse)
def get_eigen_analysis(
    image_id: str = Path(...),
    req: EigenAnalysisRequest = ...,
):
    """
    Performs eigenvalue and eigenvector analysis on a square matrix derived from image data.
    Supports Gram matrices (AAᵀ, AᵀA), sample covariance, and local non-symmetric patches.
    """
    try:
        stored = image_service.get_image(image_id)
        return eigen_service.analyze_eigenvalues(stored, req)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images/{image_id}/svd-analysis", response_model=SVDAnalysisResponse)
def get_svd_analysis(image_id: str = Path(...)):
    """
    Performs economy SVD on the grayscale image matrix: A = UΣVᵀ.
    Returns singular values, cumulative energy distribution, and effective rank.
    """
    try:
        stored = image_service.get_image(image_id)
        return svd_service.analyze_svd(stored)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/{image_id}/reconstruct", response_model=ReconstructionResponse)
def reconstruct_image(
    image_id: str = Path(...),
    req: ReconstructionRequest = ...,
):
    """
    Reconstructs image using top-k singular components via truncated SVD.
    Calculates MSE, PSNR, energy preserved, and provides side-by-side error heatmap.
    """
    try:
        stored = image_service.get_image(image_id)
        return reconstruction_service.reconstruct_image(stored, req)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

