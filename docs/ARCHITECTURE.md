# System Architecture: Mathematical Image Analysis and AI Protection System

## 1. Architectural Philosophy & Modularity

The application is structured into decoupled, modular subsystems ensuring that mathematical algorithms, image ingestion, user interface, and future protection pipelines can evolve independently without rewrites.

```
image-protection-system/
│
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI server configuration & middleware
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py            # REST API endpoints
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── interfaces.py        # Forward-compatible abstract protocols (Phases 3-8)
│   │   │   ├── image_service.py     # Image session ingestion & immutability guarantee
│   │   │   ├── matrix_service.py    # Pixel inspection, moments & numerical rank
│   │   │   ├── eigen_service.py     # Spectral decomposition (Gram, covariance, patches)
│   │   │   ├── svd_service.py       # SVD factorization & spectral energy
│   │   │   └── reconstruction_service.py # Eckart-Young TSVD reconstruction & metrics
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py           # Pydantic v2 data models & validation
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── image_utils.py       # Base64 encodings, colormaps & Pillow loaders
│   │       └── math_utils.py        # ITU-R BT.601, rank, MSE, PSNR calculations
│   ├── tests/
│   │   ├── test_services.py         # Numerical validation with small known matrices
│   │   └── test_api.py              # End-to-end API route handler verification
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/              # Scientific UI panels & Plotly wrappers
│   │   ├── services/                # Typed API client
│   │   ├── types/                   # TypeScript schemas
│   │   ├── App.tsx                  # Root state & navigation layout
│   │   ├── main.tsx
│   │   └── index.css                # Tailwind CSS v3 scientific theme
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── docs/
│   └── ARCHITECTURE.md
└── README.md
```

---

## 2. Phase 1: Image Processing Foundation

### 2.1 Immutability of Original Image
- Original raw file bytes are preserved in memory (`StoredImage.raw_bytes`).
- The image tensor $I \in \mathbb{R}^{H \times W \times 3}$ is generated once during ingestion without modifying the underlying asset.

### 2.2 Mathematical Pixel Representation
- Images are treated as 3rd-order discrete tensors $I[y, x, c] \in [0, 255]$ with row index $y \in [0, H-1]$, column index $x \in [0, W-1]$, and spectral channel index $c \in \{0, 1, 2\}$.
- The interactive pixel inspector enables neighborhood sampling through customizable windows (e.g., $(2r+1) \times (2r+1)$ submatrices), preventing memory blowup when dealing with large resolutions.

### 2.3 Transparent Grayscale Mapping
- Implements the ITU-R BT.601 photopic luma formula:
  $$Y = 0.299R + 0.587G + 0.114B$$
- Evaluated in double-precision floating-point arithmetic (`np.float64`) and displayed with transparent step-by-step arithmetic strings for educational clarity.

---

## 3. Phase 2: Mathematical Image Analysis

### 3.1 Numerical Matrix Rank & Tolerance
- Grayscale image matrix $A \in \mathbb{R}^{H \times W}$ has mathematical rank:
  $$\operatorname{rank}(A) = \sum_{i=1}^{\min(H, W)} [\sigma_i > \text{tol}]$$
  where tolerance is calculated via the standard LAPACK/NumPy criterion:
  $$\text{tol} = \sigma_1 \cdot \max(H, W) \cdot \varepsilon_{\text{machine}}$$
- **Critical Distinction:** Rank measures the global dimension of the subspace spanned by image row/column vectors (linear independence across scanlines). It does **not** identify individual important pixels or local spatial saliency.

### 3.2 Spectral Analysis ($Av = \lambda v$) vs Singular Values vs PCA
- Rectangular matrices ($H \neq W$) do not have a conventional eigenvalue decomposition because they map between spaces of different dimensionality ($\mathbb{R}^W \to \mathbb{R}^H$).
- Suitable square operators are provided:
  1. **Row Gram Matrix ($A A^T \in \mathbb{R}^{H \times H}$):** Symmetric positive semi-definite; eigenvalues equal squared singular values $\lambda_i(A A^T) = \sigma_i^2(A)$.
  2. **Column Gram Matrix ($A^T A \in \mathbb{R}^{W \times W}$):** Eigenvalues satisfy $\lambda_i(A^T A) = \sigma_i^2(A)$.
  3. **Centered Covariance Matrix:** $C = \frac{1}{N-1}(A - \mu)^T(A - \mu)$, linking directly to PCA variance axes.
  4. **Local Submatrix Patches ($P \in \mathbb{R}^{S \times S}$):** Because natural photographic patches are generally non-symmetric ($P \neq P^T$), eigenvalues can be complex ($\lambda = \alpha \pm i\beta$), representing rotational/shearing components.

### 3.3 Singular Value Decomposition (SVD)
- Decomposes matrix $A \in \mathbb{R}^{H \times W}$ into:
  $$A = U \Sigma V^T$$
  where $U \in \mathbb{R}^{H \times r}$ and $V \in \mathbb{R}^{W \times r}$ have orthonormal columns, and $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_r)$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$.
- Cumulative spectral energy captures variance concentration:
  $$E(k) = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} \times 100\%$$

### 3.4 Low-Rank Approximation (Eckart-Young-Mirsky Theorem)
- Truncated rank-$k$ reconstruction:
  $$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$$
- By the Eckart-Young-Mirsky theorem, $A_k$ is the mathematically unique global minimizer of $\|A - B\|_F$ and $\|A - B\|_2$ over all matrices $B$ with $\operatorname{rank}(B) \le k$.
- Multi-channel RGB reconstruction applies TSVD independently across color manifolds $R_k, G_k, B_k$.
- Error metrics:
  $$\text{MSE} = \frac{1}{H \cdot W} \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} (I(y, x) - \hat{I}(y, x))^2$$
  $$\text{PSNR} = 20 \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right) \quad (\text{dB})$$

---

## 4. Forward-Compatibility Roadmap (Phases 3 to 8)

The system defines clear base interfaces in `backend/app/services/interfaces.py`:

```python
class ImageTransformerInterface(ABC):
    """Phase 3: Affine transforms, shearing, scaling, rotation."""
    @abstractmethod
    def transform(self, image_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray: ...

class ImageProtectionInterface(ABC):
    """Phase 4 & 6: Mathematical perturbations, frequency cloaks & AI protection."""
    @abstractmethod
    def generate_perturbation(self, image_array: np.ndarray, config: Dict[str, Any]) -> np.ndarray: ...

    @abstractmethod
    def apply_protection(self, image_array: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]: ...

class RobustnessEvaluatorInterface(ABC):
    """Phase 5: Compression, noise, and resampling robustness analysis."""
    @abstractmethod
    def evaluate_robustness(self, original: np.ndarray, perturbed: np.ndarray, attack_params: Dict[str, Any]) -> Dict[str, float]: ...

class DeepfakeDetectorInterface(ABC):
    """Phase 7: Authenticity verification & spectral anomaly detection."""
    @abstractmethod
    def analyze_authenticity(self, image_array: np.ndarray, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
```

### Future Phase Mapping
- **Phase 3 (Geometric Transformations):** Plugs directly into `ImageTransformerInterface`. Output arrays flow immediately into existing `matrix_service`, `eigen_service`, and `svd_service` without any changes.
- **Phase 4 (Mathematical Perturbations):** Generates subtle perturbation tensors $\Delta \in \mathbb{R}^{H \times W \times C}$ added to original arrays. Existing difference heatmaps, PSNR/MSE metrics, and SVD spectra evaluate perturbation visibility and spectral footprint out of the box.
- **Phase 5 (Compression Robustness):** Compresses and decompresses perturbed arrays through JPEG/WebP pipelines and tests eigenvalue/singular value stability.
- **Phase 6 & 7 (AI Protection & Deepfake Detection):** Ingests image arrays and evaluates adversarial latent representations against generative models.
- **Phase 8 (Research Evaluation Dashboard):** Summarizes metrics across all phases into comparative benchmark tables.

