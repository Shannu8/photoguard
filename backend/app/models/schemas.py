"""Pydantic schemas and models for mathematical image analysis API."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any, Literal


class ImageMetadataResponse(BaseModel):
    image_id: str
    filename: str
    width: int
    height: int
    format: str
    channels: int
    total_pixels: int
    color_mode: str
    file_size_bytes: int
    rgb_data_url: str
    grayscale_data_url: str


class PixelInspectionRequest(BaseModel):
    x: int = Field(..., description="0-indexed X coordinate (column)")
    y: int = Field(..., description="0-indexed Y coordinate (row)")
    submatrix_radius: int = Field(default=3, description="Radius around pixel for submatrix view (default 3 means (2r+1)x(2r+1) = 7x7)")


class PixelInspectionResponse(BaseModel):
    x: int
    y: int
    r: int
    g: int
    b: int
    grayscale_exact: float
    grayscale_uint8: int
    calculation_steps: str
    submatrix_center: Tuple[int, int]
    submatrix_bounds: Dict[str, int]
    submatrix_r: List[List[int]]
    submatrix_g: List[List[int]]
    submatrix_b: List[List[int]]
    submatrix_grayscale: List[List[float]]


class ChannelStats(BaseModel):
    min: float
    max: float
    mean: float
    median: float
    std: float
    histogram: List[int]
    bin_edges: List[float]


class ImageStatisticsResponse(BaseModel):
    grayscale: ChannelStats
    red: ChannelStats
    green: ChannelStats
    blue: ChannelStats


class MatrixRankResponse(BaseModel):
    rows: int
    cols: int
    rank: int
    is_full_rank: bool
    numerical_tolerance: float
    max_possible_rank: int
    singular_values_top: List[float]
    singular_values_bottom: List[float]
    explanation: str
    scientific_notes: str


class EigenAnalysisRequest(BaseModel):
    matrix_type: Literal["gram_row", "gram_col", "covariance", "patch"] = Field(
        default="gram_row",
        description="Type of square matrix: gram_row (A A^T), gram_col (A^T A), covariance, or local patch"
    )
    patch_x: Optional[int] = Field(default=None, description="Center X for local patch")
    patch_y: Optional[int] = Field(default=None, description="Center Y for local patch")
    patch_size: int = Field(default=8, description="Patch side length (e.g. 4, 8, 16)")
    max_components: int = Field(default=50, description="Max eigenvalues/eigenvectors to return for plotting")


class EigenAnalysisResponse(BaseModel):
    matrix_type: str
    matrix_dimensions: Tuple[int, int]
    matrix_description: str
    total_eigenvalues_count: int
    eigenvalues_real: List[float]
    eigenvalues_imag: List[float]
    has_complex: bool
    sample_matrix_preview: List[List[float]]
    dominant_eigenvectors: List[List[float]]
    eigenvalues_energy_distribution: List[float]
    explanation: str
    pca_distinction: str


class SVDAnalysisResponse(BaseModel):
    rows: int
    cols: int
    total_components: int
    singular_values: List[float]
    cumulative_energy_percent: List[float]
    condition_number: float
    effective_rank: int
    explanation: str
    k_sample_points: List[int] = Field(default_factory=list)
    theoretical_psnr_curve: List[float] = Field(default_factory=list)


class ReconstructionRequest(BaseModel):
    k: int = Field(..., ge=1, description="Number of singular components to retain")
    mode: Literal["grayscale", "rgb"] = Field(default="rgb", description="Reconstruction mode: 'rgb' or 'grayscale'")


class ReconstructionResponse(BaseModel):
    k: int
    total_components: int
    retained_percentage: float
    mode: str
    mse: float
    psnr: float
    energy_preserved_percent: float
    compression_ratio: float
    original_data_url: str
    reconstructed_data_url: str
    difference_data_url: str
    explanation: str

