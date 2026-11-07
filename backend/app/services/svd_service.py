"""Singular Value Decomposition (SVD) service for matrix analysis and energy distribution."""

import numpy as np
import scipy.linalg
from typing import Dict, Any, Tuple, Optional
from app.services.image_service import StoredImage
from app.models.schemas import SVDAnalysisResponse


class SVDDataCache:
    """Caches SVD decomposition matrices U, s, Vt for fast real-time reconstruction."""
    def __init__(self, u: np.ndarray, s: np.ndarray, vt: np.ndarray):
        self.u = u
        self.s = s
        self.vt = vt


class SVDService:
    """Service performing Singular Value Decomposition on image matrices."""

    def __init__(self):
        # Cache keyed by (image_id, mode)
        self._cache: Dict[Tuple[str, str], Any] = {}

    def get_or_compute_svd(self, stored: StoredImage, mode: str = "grayscale"):
        """
        Computes or retrieves cached SVD for grayscale or RGB channels.
        Uses scipy.linalg.svd(..., full_matrices=False) for economy SVD.
        """
        cache_key = (stored.image_id, mode)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if mode == "grayscale":
            a = stored.grayscale_array
            u, s, vt = scipy.linalg.svd(a, full_matrices=False)
            data = SVDDataCache(u=u, s=s, vt=vt)
            self._cache[cache_key] = data
            return data
        elif mode == "rgb":
            # Channel-wise SVD
            rgb = stored.rgb_array
            channels_svd = {}
            for c, name in enumerate(["r", "g", "b"]):
                mat = rgb[:, :, c].astype(np.float64)
                u, s, vt = scipy.linalg.svd(mat, full_matrices=False)
                channels_svd[name] = SVDDataCache(u=u, s=s, vt=vt)
            self._cache[cache_key] = channels_svd
            return channels_svd
        else:
            raise ValueError(f"Unknown SVD mode: {mode}")

    def analyze_svd(self, stored: StoredImage) -> SVDAnalysisResponse:
        """
        Analyzes grayscale image matrix SVD and computes singular values,
        energy distributions, condition number, and effective rank.
        """
        svd_data: SVDDataCache = self.get_or_compute_svd(stored, mode="grayscale")
        s = svd_data.s
        rows, cols = stored.height, stored.width
        total_comp = len(s)

        # Singular values squared correspond to energy
        s_squared = s ** 2
        total_energy = float(np.sum(s_squared))

        if total_energy > 1e-12:
            cum_energy = (np.cumsum(s_squared) / total_energy) * 100.0
            cum_energy_list = [round(float(v), 3) for v in cum_energy]
        else:
            cum_energy_list = [0.0] * total_comp

        # Condition number = sigma_max / sigma_min
        cond = float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")

        # Effective rank to capture 99% of mathematical energy
        idx_99 = np.searchsorted(cum_energy, 99.0) if total_energy > 1e-12 else total_comp
        effective_rank = int(min(total_comp, idx_99 + 1))

        explanation = (
            f"SVD decomposes image matrix A ({rows}×{cols}) into A = UΣVᵀ, where U ({rows}×{total_comp}) "
            f"and V ({cols}×{total_comp}) have orthonormal columns, and Σ contains {total_comp} singular values in descending order. "
            f"The largest singular value is σ₁ = {s[0]:.2f}, and the condition number is κ(A) = {cond:.2e}. "
            f"Just {effective_rank} singular components ({effective_rank / total_comp * 100:.1f}% of total) "
            "account for over 99.0% of the total spectral Frobenius energy."
        )

        # Compute analytical TSVD Frobenius error and theoretical PSNR curve
        # ||A - A_k||_F^2 = sum_{i=k}^{r-1} s[i]^2
        hw = rows * cols
        rev_cumsum = np.cumsum(s_squared[::-1])[::-1]
        # rev_cumsum[k] gives tail sum from index k to end
        k_pts = []
        psnr_pts = []

        step = max(1, total_comp // 40)
        sample_indices = sorted(list(set(
            [1, 2, 3, 5, 10, 20, 30, 50, 75, 100] +
            list(range(1, total_comp + 1, step)) +
            [total_comp]
        )))
        sample_indices = [idx for idx in sample_indices if 1 <= idx <= total_comp]

        for k_val in sample_indices:
            k_pts.append(k_val)
            if k_val >= total_comp:
                psnr_pts.append(100.0)
            else:
                tail_err = rev_cumsum[k_val]
                mse_k = tail_err / hw
                if mse_k <= 1e-12:
                    psnr_pts.append(100.0)
                else:
                    psnr_pts.append(round(float(20.0 * np.log10(255.0 / np.sqrt(mse_k))), 2))

        return SVDAnalysisResponse(
            rows=rows,
            cols=cols,
            total_components=total_comp,
            singular_values=[round(float(val), 4) for val in s],
            cumulative_energy_percent=cum_energy_list,
            condition_number=cond,
            effective_rank=effective_rank,
            explanation=explanation,
            k_sample_points=k_pts,
            theoretical_psnr_curve=psnr_pts,
        )


svd_service = SVDService()

