# Mathematical Image Analysis and AI Image Protection System (PhotoGuard)

An educational and university-level mathematical computing platform for analyzing images as discrete tensors and matrices, evaluating spectral decompositions (eigenvalues and eigenvectors), performing Singular Value Decomposition (SVD), and reconstructing low-rank approximations via the Eckart-Young-Mirsky theorem.

Built strictly in accordance with **Phase 1 (Image Processing Foundation)** and **Phase 2 (Mathematical Image Analysis)** requirements, with forward-compatible interfaces designed for subsequent research phases (transformations, perturbations, robustness testing, AI protection, and deepfake detection).

---

## 🚀 Key Features

### Phase 1 — Image Processing Foundation
- **Image Ingestion & Immutability:** Ingests PNG, JPEG/JPG, WebP, and BMP files. Displays dimensions, channel count, format, and total pixel count. The original raw file bytes are preserved without modification.
- **Mathematical RGB Representation:** Represents images as $I \in \mathbb{R}^{H \times W \times 3}$.
- **Interactive Pixel Inspector:** Click any coordinate $(x, y)$ or input coordinates directly to inspect $R, G, B$ values, exact photopic grayscale luminance, and view $(2r+1) \times (2r+1)$ local submatrix neighborhood tables/heatmaps.
- **Transparent RGB-to-Grayscale:** Evaluates the ITU-R BT.601 standard formula:
  $$Y = 0.299R + 0.587G + 0.114B$$
  with step-by-step arithmetic breakdowns and side-by-side or toggled image comparisons.
- **Statistical Moments & Histograms:** Computes minimum, maximum, mean ($\mu$), median, and standard deviation ($\sigma$) alongside interactive 256-bin Plotly histograms across Red, Green, Blue, and Grayscale channels with linear/log scale toggle.

### Phase 2 — Mathematical Image Analysis
- **Numerical Matrix Rank & Linear Independence:** Calculates $\operatorname{rank}(A)$ of the grayscale matrix using SVD and reports whether the matrix is full rank ($\operatorname{rank} = \min(H, W)$) or rank-deficient. Explains the numerical tolerance cutoff $\text{tol} = \sigma_1 \cdot \max(H, W) \cdot \varepsilon$.
  - *Educational Principle:* Clarifies that matrix rank measures global algebraic subspace dimension across scanlines, not individual pixel saliency.
- **Eigenvalue and Eigenvector Analysis ($Av = \lambda v$):** 
  - Resolves the mathematical requirement that rectangular matrices ($H \neq W$) do not have conventional eigenvalue decompositions.
  - Supports analyzing **Row Gram Matrix ($A A^T$)**, **Column Gram Matrix ($A^T A$)**, **Centered Covariance Matrix ($C$)**, and **Local Image Patches ($P \in \mathbb{R}^{S \times S}$)**.
  - Detects complex eigenvalues ($\lambda = \alpha \pm i\beta$) that arise naturally from non-symmetric photographic image patches.
  - Clearly distinguishes eigenvalue analysis from Principal Component Analysis (PCA).
- **Singular Value Decomposition ($A = U \Sigma V^T$):** Computes singular values, condition number $\kappa(A) = \sigma_{\max} / \sigma_{\min}$, effective rank (99% spectral energy), scree plot decay curves, and cumulative energy distribution $E(k)$.
- **Low-Rank Reconstruction (TSVD):** Reconstructs images using top-$k$ components via:
  $$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$$
  - Supports both **Multi-Channel RGB (channel-wise SVD)** and **Grayscale Luma Matrix SVD**.
  - Side-by-side comparative views: Original vs Reconstructed ($A_k$) vs Absolute Error Heatmap ($|A - A_k|$ in Inferno colormap).
  - Quantitative metrics: Retained components ($k$ and %), Spectral Energy Preserved (%), Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR in dB), and degrees-of-freedom compression ratio.
  - Qualitative visual interpretations explaining frequency filtering, blur, and edge resolution at varying $k$.

---

## 🛠️ Technology Stack

- **Backend:** Python 3.11, FastAPI, NumPy, SciPy, Pillow, OpenCV, Pytest
- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS v3, Lucide Icons
- **Visualization:** Plotly.js (`plotly.js-dist-min`)

---

## 📦 Installation & Execution

### 1. Backend Setup
From the repository root:

```bash
cd backend

# (Optional) Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API will be live at `http://localhost:8000` (Interactive Swagger docs at `http://localhost:8000/docs`).

### 2. Frontend Setup
In a new terminal window:

```bash
cd frontend

# Install npm dependencies
npm install

# Start Vite development server
npm run dev
```
The application will be accessible in your browser at `http://localhost:5173`.

---

## 🧪 Mathematical Accuracy & Unit Tests

The backend includes test suites verifying numerical accuracy against known mathematical matrices (identity matrices, rank-deficient outer products, symmetric 2x2 eigen systems, 90° rotation complex eigenvalues, SVD orthonormality $U^TU = I$, Eckart-Young monotonicity, and ITU-R BT.601 exact conversions).

Run the tests using pytest:

```bash
# From workspace root
$env:PYTHONPATH='backend'; python -m pytest backend/tests -v
```

All 9 test suites pass:
```text
backend/tests/test_api.py::test_api_handlers_pipeline PASSED
backend/tests/test_services.py::test_rgb_to_grayscale_known_values PASSED
backend/tests/test_services.py::test_pixel_calculation_formatting PASSED
backend/tests/test_services.py::test_mse_and_psnr PASSED
backend/tests/test_services.py::test_matrix_rank PASSED
backend/tests/test_services.py::test_eigenvalue_known_symmetric_matrix PASSED
backend/tests/test_services.py::test_complex_eigenvalues_non_symmetric PASSED
backend/tests/test_services.py::test_svd_orthonormality_and_reconstruction PASSED
backend/tests/test_services.py::test_channel_statistics PASSED
```

To verify the frontend production build:

```bash
cd frontend
npm run build
```

---

## 🔮 Future Phase Compatibility (Phases 3 to 8)

The codebase has abstract interfaces (`backend/app/services/interfaces.py`) ensuring zero rewriting is required when implementing future phases:
- **Phase 3:** Image transformations (rotation, scaling, reflection, shearing via `ImageTransformerInterface`).
- **Phase 4:** Mathematical image-protection perturbations (via `ImageProtectionInterface`).
- **Phase 5:** Compression and resizing robustness analysis (via `RobustnessEvaluatorInterface`).
- **Phase 6:** AI manipulation protection experiments.
- **Phase 7:** Deepfake detection and spectral anomaly scoring (via `DeepfakeDetectorInterface`).
- **Phase 8:** Final research dashboard and automated evaluation benchmarks.

