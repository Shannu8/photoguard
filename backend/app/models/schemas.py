"""Pydantic schemas and models for mathematical image analysis API."""

from pydantic import BaseModel, Field, model_validator
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


# ==============================================================================
# Phase 3: Mathematical Transformations & Tensor Export Schemas
# ==============================================================================

class TransformationStep(BaseModel):
    type: Literal[
        "rotate",
        "scale",
        "reflect",
        "shear",
        "translate",
        "random_affine",
        "resize",
        "jpeg_compression",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)


class TransformationRequest(BaseModel):
    base_source: Literal["original", "perturbed"] = Field(
        default="original",
        description="Target image to transform: 'original' uploaded image or Phase 4 'perturbed' image"
    )
    operations: List[TransformationStep] = Field(..., description="Ordered list of transformation operations")
    interpolation: Literal["bilinear", "nearest", "bicubic"] = Field(default="bilinear")
    canvas_mode: Literal["fit", "same"] = Field(default="fit", description="'fit' expands canvas to prevent cropping; 'same' keeps input dimensions")
    selected_point: Optional[Tuple[float, float]] = Field(default=None, description="Optional (x, y) coordinate to map through transformation")


class MatrixStepInfo(BaseModel):
    name: str
    operation_type: str
    matrix_3x3: List[List[float]]
    formula_latex: str
    description: str


class TransformationResponse(BaseModel):
    original_width: int
    original_height: int
    transformed_width: int
    transformed_height: int
    composite_matrix_3x3: List[List[float]]
    steps: List[MatrixStepInfo]
    matrix_multiplication_explanation: str
    original_point: Optional[Tuple[float, float]] = None
    transformed_point: Optional[Tuple[float, float]] = None
    transformed_rgb_data_url: str
    transformed_grayscale_data_url: str
    interpolation: str
    canvas_mode: str
    determinant: float
    is_invertible: bool


class TensorInfo(BaseModel):
    name: str
    shape: List[int]
    dtype: str
    total_elements: int
    estimated_size_bytes: int


class TensorMetadataResponse(BaseModel):
    has_transformed: bool
    original_rgb: TensorInfo
    original_grayscale: TensorInfo
    transformed_rgb: Optional[TensorInfo] = None
    transformed_grayscale: Optional[TensorInfo] = None
    has_perturbed: bool = False
    perturbed_rgb: Optional[TensorInfo] = None
    perturbed_grayscale: Optional[TensorInfo] = None
    perturbation_delta: Optional[TensorInfo] = None


# ==============================================================================
# Phase 4: Mathematical Image Perturbation Engine Schemas
# ==============================================================================

PerturbationMethod = Literal[
    "additive",
    "multiplicative",
    "gaussian",
    "salt_and_pepper",
    "inversion",
    "brightness_contrast",
]

CompositionMode = Literal["sequential", "weighted"]


class BudgetConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether budget projection is active")
    type: Literal["none", "linf", "l2", "both"] = Field(
        default="none",
        description="Budget projection constraint: 'none', 'linf', 'l2', or 'both'"
    )
    epsilon_inf: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="L-infinity budget bound ||Δ||_∞ <= ε_inf (default 8/255 ≈ 0.03137 in normalized [0,1] units)"
    )
    epsilon_linf: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Backward-compatible alias for epsilon_inf"
    )
    epsilon_2: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="L2 budget bound ||Δ||_2 <= ε_2 across entire image tensor"
    )
    epsilon_l2: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Backward-compatible alias for epsilon_2"
    )
    l0_threshold: Optional[float] = Field(
        default=1e-4,
        ge=0.0,
        description="Threshold τ for L0 norm element and pixel detection (normalized [0,1] or pixel scale)"
    )
    max_changed_pixel_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional maximum ratio of changed pixels allowed [0.0, 1.0]"
    )
    norm_scope: Literal["entire_tensor", "per_channel", "per_pixel"] = Field(
        default="entire_tensor",
        description="Scope of norm calculation (default: entire_tensor)"
    )
    unit: Literal["normalized_0_1", "pixel_0_255"] = Field(
        default="normalized_0_1",
        description="Explicit unit for budget parameters: 'normalized_0_1' or 'pixel_0_255'"
    )

    @model_validator(mode="after")
    def sync_and_validate_budget(self) -> "BudgetConfig":
        # Synchronize epsilon_inf and epsilon_linf
        if self.epsilon_inf is None and self.epsilon_linf is not None:
            self.epsilon_inf = self.epsilon_linf
        elif self.epsilon_inf is not None and self.epsilon_linf is None:
            self.epsilon_linf = self.epsilon_inf

        # Default epsilon_inf for 'linf' and 'both' if not specified
        if self.type in ("linf", "both") and self.epsilon_inf is None:
            self.epsilon_inf = (8.0 / 255.0) if self.unit == "normalized_0_1" else 8.0
            self.epsilon_linf = self.epsilon_inf

        # Synchronize epsilon_2 and epsilon_l2
        if self.epsilon_2 is None and self.epsilon_l2 is not None:
            self.epsilon_2 = self.epsilon_l2
        elif self.epsilon_2 is not None and self.epsilon_l2 is None:
            self.epsilon_l2 = self.epsilon_2

        # In normalized_0_1 mode, L-infinity epsilon cannot exceed 1.0 (maximum dynamic range [0, 1])
        # Do not silently interpret values > 1 as 8-bit values!
        if self.unit == "normalized_0_1" and self.epsilon_inf is not None:
            if self.epsilon_inf > 1.0:
                raise ValueError(
                    f"epsilon_inf={self.epsilon_inf} exceeds 1.0 for unit='normalized_0_1'. "
                    "Normalized values must be in [0.0, 1.0]. If specifying 8-bit pixel values (0-255), "
                    "explicitly set unit='pixel_0_255'."
                )

        # Validate presence of constraints when enabled
        if self.enabled and self.type in ("linf", "both"):
            if self.epsilon_inf is None or self.epsilon_inf < 0:
                raise ValueError("epsilon_inf must be specified and non-negative when type is 'linf' or 'both'")
        if self.enabled and self.type in ("l2", "both"):
            if self.epsilon_2 is None or self.epsilon_2 < 0:
                raise ValueError("epsilon_2 must be specified and non-negative when type is 'l2' or 'both'")

        return self


class PerturbationStep(BaseModel):
    id: Optional[str] = None
    type: PerturbationMethod
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=5.0, description="Weight (0.0 to 5.0) for weighted perturbation combination")
    strength: Optional[float] = Field(default=None, ge=0.0, description="Optional unified strength magnitude")
    parameter_unit: Literal["normalized_0_1", "pixel_0_255"] = Field(
        default="pixel_0_255",
        description="Explicit unit for operator parameters"
    )
    seed: Optional[int] = Field(default=None, ge=0, description="Optional individual step pseudorandom seed")
    channels: List[Literal["R", "G", "B"]] = Field(default=["R", "G", "B"])
    params: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional operation metadata")

    @model_validator(mode="after")
    def validate_step(self) -> "PerturbationStep":
        if not self.channels:
            raise ValueError("channels list cannot be empty; must select at least one of 'R', 'G', 'B'")

        # Check normalized parameter boundaries without silent interpretation
        if self.parameter_unit == "normalized_0_1":
            if self.strength is not None and self.strength > 1.0:
                raise ValueError(
                    f"strength={self.strength} exceeds 1.0 for parameter_unit='normalized_0_1'. "
                    "Normalized parameters must be in [0.0, 1.0]. Use parameter_unit='pixel_0_255' for 8-bit scales."
                )
            if self.type == "additive":
                eps = self.params.get("epsilon")
                if eps is not None and float(eps) > 1.0:
                    raise ValueError(
                        f"Additive epsilon={eps} exceeds 1.0 for parameter_unit='normalized_0_1'. "
                        "Set parameter_unit='pixel_0_255' for pixel scale [0, 255]."
                    )
            elif self.type == "gaussian":
                sigma = self.params.get("sigma")
                if sigma is not None and float(sigma) > 1.0:
                    raise ValueError(
                        f"Gaussian sigma={sigma} exceeds 1.0 for parameter_unit='normalized_0_1'. "
                        "Set parameter_unit='pixel_0_255' for pixel scale [0, 255]."
                    )

        # Validate method-specific parameter ranges
        if self.type == "salt_and_pepper":
            p = self.params.get("salt_pepper_prob")
            if p is not None and (float(p) < 0.0 or float(p) > 1.0):
                raise ValueError(f"salt_pepper_prob must be between 0.0 and 1.0, got {p}")
            r = self.params.get("salt_vs_pepper_ratio")
            if r is not None and (float(r) < 0.0 or float(r) > 1.0):
                raise ValueError(f"salt_vs_pepper_ratio must be between 0.0 and 1.0, got {r}")
        elif self.type == "multiplicative":
            factor = self.params.get("multiplicative_factor")
            if factor is not None and (float(factor) < -1.0 or float(factor) > 2.0):
                raise ValueError(f"multiplicative_factor must be between -1.0 and 2.0, got {factor}")
        elif self.type == "brightness_contrast":
            c = self.params.get("contrast_factor")
            if c is not None and float(c) < 0.0:
                raise ValueError(f"contrast_factor must be non-negative, got {c}")

        return self


class PerturbationConfig(BaseModel):
    version: str = Field(default="4.0", description="Configuration schema version")
    composition_mode: CompositionMode = Field(
        default="sequential",
        description="Composition mode: 'sequential' (Markov chain) or 'weighted' (superposition)"
    )
    base_source: Literal["original", "transformed"] = Field(
        default="original",
        description="Target image to perturb: 'original' or 'transformed'"
    )
    budget: Optional[BudgetConfig] = Field(
        default=None,
        description="Perturbation budget configuration"
    )
    seed: Optional[int] = Field(
        default=42,
        ge=0,
        description="Pseudorandom generator seed for deterministic reproducibility"
    )
    operations: List[PerturbationStep] = Field(
        default_factory=list,
        description="Ordered list of perturbation operations"
    )
    l0_threshold: float = Field(
        default=1e-4,
        ge=0.0,
        description="Configurable threshold τ for L0 norm element and pixel detection"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional user experiment metadata"
    )


class PerturbationRequest(BaseModel):
    # Optional nested config
    config: Optional[PerturbationConfig] = Field(
        default=None,
        description="Complete serialized perturbation configuration"
    )

    # Composition pipeline fields
    composition_mode: CompositionMode = Field(
        default="sequential",
        description="Composition mode: 'sequential' (Markov chain) or 'weighted' (superposition)"
    )
    budget: Optional[BudgetConfig] = Field(
        default=None,
        description="Formal perturbation budget projection module (L-infinity, L2 ball projection)"
    )
    global_bound: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=255.0,
        description="Legacy global perturbation bound ||Δ_combined||_∞ <= ε_global for backward compatibility"
    )
    operations: Optional[List[PerturbationStep]] = Field(
        default=None,
        description="Ordered list of perturbation operations for pipeline execution"
    )
    l0_threshold: float = Field(
        default=1e-4,
        ge=0.0,
        description="Configurable threshold τ for L0 norm element and pixel detection (e.g. 1/255 ≈ 0.00392 or 1e-4)"
    )

    # Backward-compatible single-step fields
    method: Optional[PerturbationMethod] = Field(default=None, description="Perturbation technique (for single-step invocation)")
    base_source: Literal["original", "transformed"] = Field(
        default="original",
        description="Target image to perturb: 'original' uploaded image or Phase 3 'transformed' image"
    )
    channels: List[Literal["R", "G", "B"]] = Field(
        default=["R", "G", "B"],
        description="Channels to perturb. Pixels in unselected channels remain unchanged."
    )
    epsilon: Optional[float] = Field(default=10.0, ge=0.0, le=255.0, description="Max perturbation bound ||Δ||_∞ <= ε")
    additive_mode: Literal["random", "uniform", "positive_const", "negative_const"] = Field(
        default="random",
        description="Additive distribution mode: uniform, random integer, or directional constant"
    )
    multiplicative_factor: Optional[float] = Field(
        default=0.1, ge=-1.0, le=2.0, description="Multiplicative strength factor (e.g. 0.1 for ±10%)"
    )
    sigma: Optional[float] = Field(default=15.0, ge=0.0, le=100.0, description="Gaussian noise standard deviation σ")
    salt_pepper_prob: Optional[float] = Field(
        default=0.05, ge=0.0, le=1.0, description="Probability of pixel corruption p"
    )
    salt_vs_pepper_ratio: Optional[float] = Field(
        default=0.5, ge=0.0, le=1.0, description="Proportion of salt (255) vs pepper (0)"
    )
    contrast_factor: Optional[float] = Field(
        default=1.2, ge=0.1, le=3.0, description="Contrast factor c (1.0 = neutral)"
    )
    brightness_offset: Optional[float] = Field(
        default=10.0, ge=-100.0, le=100.0, description="Brightness bias b (0.0 = neutral)"
    )
    seed: Optional[int] = Field(default=42, description="Pseudorandom generator seed for deterministic reproducibility")
    selected_point: Optional[Tuple[int, int]] = Field(default=None, description="Optional (x, y) coordinates for local pixel inspection")
    submatrix_radius: int = Field(default=2, ge=1, le=10, description="Radius for local perturbation neighborhood matrix")

    @model_validator(mode="before")
    @classmethod
    def unpack_nested_config(cls, data: Any) -> Any:
        if isinstance(data, dict) and "config" in data and isinstance(data["config"], dict):
            cfg = data["config"]
            for key in ["composition_mode", "budget", "operations", "seed", "base_source", "l0_threshold"]:
                if key in cfg and key not in data:
                    data[key] = cfg[key]
        return data


class ChannelMetrics(BaseModel):
    channel: str
    l1_norm: float
    l2_norm: float
    linf_norm: float
    mae: float
    mse: float
    mean_diff: float
    std_diff: float
    min_diff: float
    max_diff: float
    changed_count: int


class PerturbationMetrics(BaseModel):
    mse: float
    psnr: float
    ssim: float
    mae: float
    l1_norm: float
    l2_norm: float
    linf_norm: float
    max_abs_perturbation: float
    min_difference: float
    max_difference: float
    mean_difference: float
    std_diff: float = 0.0
    l0_threshold: float = 1e-4
    changed_elements_count: int = 0
    total_elements_count: int = 0
    changed_elements_percent: float = 0.0
    changed_pixels_count: int = 0
    total_pixels_count: int = 0
    changed_pixels_percent: float = 0.0
    modified_pixels_count: int
    modified_pixels_percent: float
    lpips: Optional[float] = None
    per_channel: Optional[Dict[str, ChannelMetrics]] = None
    quantization_distortion: Optional[Dict[str, float]] = None


class IntermediatePixelValue(BaseModel):
    step_index: int
    step_name: str
    rgb: List[int]
    delta_from_previous: List[float]
    delta_from_original: List[float]


class PerturbationPixelInspection(BaseModel):
    x: int
    y: int
    original_rgb: List[int]
    intermediate_pixels: List[IntermediatePixelValue] = Field(default_factory=list)
    perturbed_rgb: List[int]
    delta_rgb: List[float]
    abs_delta_rgb: List[float]
    original_gray: float
    perturbed_gray: float
    delta_gray: float
    submatrix_radius: int
    submatrix_bounds: Dict[str, int]
    submatrix_delta_r: List[List[float]]
    submatrix_delta_g: List[List[float]]
    submatrix_delta_b: List[List[float]]
    submatrix_delta_gray: List[List[float]]


class PerturbationStepResult(BaseModel):
    step_index: int
    step_id: str
    type: str
    name: str
    category: Literal["bounded_noise", "educational_intensity", "photometric"]
    enabled: bool
    weight: float
    formula_latex: str
    preview_data_url: str
    intermediate_metrics: PerturbationMetrics


class PerturbationResponse(BaseModel):
    image_id: str
    method: str
    composition_mode: str = "sequential"
    base_source: str
    width: int
    height: int
    channels: int
    global_bound: Optional[float] = None
    budget_report: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any]
    reproducibility_seed: Optional[int] = None
    original_data_url: str
    perturbed_rgb_data_url: str
    perturbed_grayscale_data_url: str
    difference_heatmap_data_url: str
    signed_difference_data_url: str
    metrics: PerturbationMetrics
    pixel_inspection: Optional[PerturbationPixelInspection] = None
    intermediate_steps: List[PerturbationStepResult] = Field(default_factory=list)
    method_explanation: str
    mathematical_formula: str



