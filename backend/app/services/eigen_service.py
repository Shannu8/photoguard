"""Eigenvalue and eigenvector analysis service with rigorous mathematical foundation."""

import numpy as np
import scipy.linalg
from typing import Dict, Any, Tuple, List, Optional
from app.services.image_service import StoredImage
from app.models.schemas import EigenAnalysisRequest, EigenAnalysisResponse


class EigenService:
    """
    Performs eigenvalue and eigenvector analysis on square image-derived matrices.
    Provides clear mathematical explanations regarding rectangular vs square matrices,
    Gram matrices, covariance matrices, and general non-symmetric patches.
    """

    @staticmethod
    def analyze_eigenvalues(
        stored: StoredImage, req: EigenAnalysisRequest
    ) -> EigenAnalysisResponse:
        gray = stored.grayscale_array
        h, w = gray.shape

        # Select square matrix based on requested mode
        if req.matrix_type == "patch":
            # Extract square patch around center (or image center if unspecified)
            px = req.patch_x if req.patch_x is not None else w // 2
            py = req.patch_y if req.patch_y is not None else h // 2
            size = min(req.patch_size, 32)  # Cap patch size for performance & clarity
            r_start = max(0, min(h - size, py - size // 2))
            c_start = max(0, min(w - size, px - size // 2))
            target_matrix = gray[r_start : r_start + size, c_start : c_start + size].copy()

            matrix_desc = (
                f"Local {size}×{size} Image Submatrix Patch extracted at region "
                f"rows [{r_start}:{r_start+size}], cols [{c_start}:{c_start+size}]. "
                "Because natural image patches are generally non-symmetric (A ≠ Aᵀ), "
                "eigenvalues may be complex numbers (λ = α + iβ) appearing in conjugate pairs."
            )

        elif req.matrix_type == "covariance":
            # Feature covariance across columns
            # If large, subsample rows to prevent excessive computation
            stride = max(1, h // 256)
            sub_gray = gray[::stride, :]
            mean_centered = sub_gray - np.mean(sub_gray, axis=0, keepdims=True)
            cov_mat = (mean_centered.T @ mean_centered) / max(1, sub_gray.shape[0] - 1)
            target_matrix = cov_mat

            matrix_desc = (
                f"Sample Covariance Matrix ({target_matrix.shape[0]}×{target_matrix.shape[1]}), "
                "computed as C = 1/(N-1) * (A - μ)ᵀ(A - μ). "
                "This matrix is symmetric positive semi-definite; all eigenvalues are real and non-negative."
            )

        elif req.matrix_type == "gram_col":
            # Column Gram matrix A^T A
            stride = max(1, h // 256)
            sub_gray = gray[::stride, :]
            target_matrix = sub_gray.T @ sub_gray

            matrix_desc = (
                f"Column Gram / Inner-Product Matrix AᵀA ({target_matrix.shape[0]}×{target_matrix.shape[1]}). "
                "Symmetric positive semi-definite. Mathematically, the eigenvalues of AᵀA are exactly "
                "the squares of the singular values of A: λᵢ(AᵀA) = σᵢ(A)²."
            )

        else:  # "gram_row" default
            # Row Gram matrix A A^T
            stride = max(1, w // 256)
            sub_gray = gray[:, ::stride]
            target_matrix = sub_gray @ sub_gray.T

            matrix_desc = (
                f"Row Gram / Inner-Product Matrix AAᵀ ({target_matrix.shape[0]}×{target_matrix.shape[1]}). "
                "Symmetric positive semi-definite matrix encoding pairwise row correlation. "
                "Eigenvalues are real and correspond to squared singular values: λᵢ(AAᵀ) = σᵢ(A)²."
            )

        # For very large target matrices, keep max dimension manageable for browser transfer
        n_dim = target_matrix.shape[0]
        if n_dim > 128 and req.matrix_type != "patch":
            # Subsample square matrix for numerical stability and prompt display
            step = int(np.ceil(n_dim / 128))
            analysis_mat = target_matrix[::step, ::step].copy()
        else:
            analysis_mat = target_matrix.copy()

        # Compute eigenvalues and eigenvectors using scipy.linalg.eig
        # Handles both symmetric and non-symmetric (complex) cases
        eigvals, eigvecs = scipy.linalg.eig(analysis_mat)

        # Sort by eigenvalue magnitude |λ| descending
        magnitudes = np.abs(eigvals)
        sort_indices = np.argsort(magnitudes)[::-1]
        sorted_eigvals = eigvals[sort_indices]
        sorted_eigvecs = eigvecs[:, sort_indices]

        has_complex = bool(np.any(np.abs(np.imag(sorted_eigvals)) > 1e-6))

        # Format eigenvalues for response (limit to max_components)
        max_comp = min(req.max_components, len(sorted_eigvals))
        real_parts = [round(float(np.real(v)), 5) for v in sorted_eigvals[:max_comp]]
        imag_parts = [round(float(np.imag(v)), 5) for v in sorted_eigvals[:max_comp]]

        # Normalized energy distribution (magnitude of eigenvalues)
        total_mag = float(np.sum(magnitudes))
        if total_mag > 1e-12:
            energy_dist = [round(float(m / total_mag) * 100.0, 4) for m in magnitudes[:max_comp]]
        else:
            energy_dist = [0.0] * max_comp

        # Preview sample matrix (top 8x8 slice)
        sample_preview_dim = min(8, analysis_mat.shape[0])
        sample_preview = np.round(analysis_mat[:sample_preview_dim, :sample_preview_dim], 3).tolist()

        # Dominant eigenvectors (real part of top 5 eigenvectors, sampled)
        top_k_vecs = min(5, sorted_eigvecs.shape[1])
        vec_sample_len = min(16, sorted_eigvecs.shape[0])
        dominant_vecs = [
            [round(float(np.real(v)), 4) for v in sorted_eigvecs[:vec_sample_len, i]]
            for i in range(top_k_vecs)
        ]

        explanation = (
            f"Analyzed {analysis_mat.shape[0]}×{analysis_mat.shape[1]} square matrix derived from image data. "
            f"The characteristic equation det(A - λI) = 0 yields {len(sorted_eigvals)} eigenvalues. "
            f"{'Complex eigenvalues were detected, representing rotational/shearing components in the non-symmetric patch.' if has_complex else 'All eigenvalues are real, consistent with a symmetric positive semi-definite matrix.'} "
            f"Eigenvectors v satisfy Av = λv, representing directions that undergo pure scaling by λ without changing orientation."
        )

        pca_distinction = (
            "MATHEMATICAL DISTINCTION: EIGENVALUES vs PCA vs SVD: "
            "1. Rectangular Images: Rectangular matrices A ∈ R^{H×W} (H ≠ W) map vectors between different dimensional spaces (R^W → R^H). "
            "Thus, conventional eigenvalues Av = λv are geometrically undefined for rectangular images without forming a square matrix. "
            "2. Gram Matrix (AᵀA, AAᵀ): Eigenvalues of AᵀA or AAᵀ equal the squared singular values of A: λᵢ = σᵢ². "
            "Eigenvectors of AAᵀ are the left singular vectors U; eigenvectors of AᵀA are the right singular vectors V. "
            "3. PCA: PCA computes eigenvalues of the centered covariance matrix 1/(N-1)(A - μ)ᵀ(A - μ), capturing variance directions. "
            "Uncentered SVD captures total root-mean-square energy rather than variance around the mean."
        )

        return EigenAnalysisResponse(
            matrix_type=req.matrix_type,
            matrix_dimensions=(analysis_mat.shape[0], analysis_mat.shape[1]),
            matrix_description=matrix_desc,
            total_eigenvalues_count=len(sorted_eigvals),
            eigenvalues_real=real_parts,
            eigenvalues_imag=imag_parts,
            has_complex=has_complex,
            sample_matrix_preview=sample_preview,
            dominant_eigenvectors=dominant_vecs,
            eigenvalues_energy_distribution=energy_dist,
            explanation=explanation,
            pca_distinction=pca_distinction,
        )


eigen_service = EigenService()

