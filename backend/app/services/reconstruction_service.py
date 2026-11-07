"""Image reconstruction service using Truncated Singular Value Decomposition (TSVD)."""

import numpy as np
from typing import Dict, Any, Tuple
from app.services.image_service import StoredImage
from app.services.svd_service import svd_service, SVDDataCache
from app.utils.math_utils import calculate_mse, calculate_psnr
from app.utils.image_utils import array_to_base64_png, generate_error_heatmap_base64
from app.models.schemas import ReconstructionRequest, ReconstructionResponse


class ReconstructionService:
    """Service handling low-rank matrix approximation and reconstruction metrics."""

    @staticmethod
    def reconstruct_image(
        stored: StoredImage, req: ReconstructionRequest
    ) -> ReconstructionResponse:
        """
        Reconstructs image using top-k singular components via Eckart-Young-Mirsky approximation.
        Supports both RGB (channel-wise) and Grayscale modes.
        """
        rows, cols = stored.height, stored.width
        max_k = min(rows, cols)
        k = max(1, min(max_k, req.k))

        if req.mode == "grayscale":
            svd_data: SVDDataCache = svd_service.get_or_compute_svd(stored, mode="grayscale")
            u = svd_data.u[:, :k]
            s = svd_data.s[:k]
            vt = svd_data.vt[:k, :]

            # Truncated reconstruction: A_k = U_k * diag(s_k) * V_k^T
            recon_mat = (u * s) @ vt
            clipped_recon = np.clip(recon_mat, 0, 255)

            # Error metrics
            orig_mat = stored.grayscale_array
            mse = calculate_mse(orig_mat, clipped_recon)
            psnr = calculate_psnr(orig_mat, clipped_recon)

            total_energy = np.sum(svd_data.s ** 2)
            k_energy = np.sum(s ** 2)
            energy_percent = float((k_energy / total_energy) * 100.0) if total_energy > 0 else 100.0

            reconstructed_url = array_to_base64_png(clipped_recon, is_grayscale=True)
            diff_url = generate_error_heatmap_base64(orig_mat, clipped_recon)
            original_url = stored.grayscale_data_url

        else:  # RGB channel-wise reconstruction
            channels_svd = svd_service.get_or_compute_svd(stored, mode="rgb")
            recon_channels = []
            total_k_energy = 0.0
            total_all_energy = 0.0

            for c_name in ["r", "g", "b"]:
                c_data: SVDDataCache = channels_svd[c_name]
                u = c_data.u[:, :k]
                s = c_data.s[:k]
                vt = c_data.vt[:k, :]

                recon_c = (u * s) @ vt
                recon_channels.append(np.clip(recon_c, 0, 255))

                total_k_energy += np.sum(s ** 2)
                total_all_energy += np.sum(c_data.s ** 2)

            recon_rgb = np.stack(recon_channels, axis=2)
            orig_rgb = stored.rgb_array

            mse = calculate_mse(orig_rgb, recon_rgb)
            psnr = calculate_psnr(orig_rgb, recon_rgb)
            energy_percent = (
                float((total_k_energy / total_all_energy) * 100.0)
                if total_all_energy > 0
                else 100.0
            )

            reconstructed_url = array_to_base64_png(recon_rgb, is_grayscale=False)
            diff_url = generate_error_heatmap_base64(orig_rgb, recon_rgb)
            original_url = stored.rgb_data_url

        # Compression degrees of freedom ratio
        # Truncated SVD requires k * (H + W + 1) floats vs H * W raw values
        params_needed = k * (rows + cols + 1)
        raw_params = rows * cols
        compression_ratio = round(float(raw_params / params_needed), 2)
        retained_pct = round(float((k / max_k) * 100.0), 2)

        # Visual differences explanation based on k
        if k <= 5:
            visual_desc = (
                f"At k={k} components ({retained_pct}% of rank), only gross luminance gradients and dominant "
                "horizontal/vertical outer products are captured. Fine structural edges are absent and high-frequency "
                "details are completely filtered out."
            )
        elif k <= 25:
            visual_desc = (
                f"At k={k} components ({retained_pct}% of rank), prominent object silhouettes and major tonal contours "
                "are clearly recognizable. Fine textures, high-frequency noise, and sharp boundaries remain softened."
            )
        elif k <= 60:
            visual_desc = (
                f"At k={k} components ({retained_pct}% of rank), the image captures {energy_percent:.1f}% of spectral energy. "
                "Edges are sharp and secondary textures are restored with high visual fidelity."
            )
        else:
            visual_desc = (
                f"At k={k} components ({retained_pct}% of rank), reconstruction achieves PSNR {psnr:.2f} dB and "
                f"{energy_percent:.2f}% energy preservation. The reconstruction is nearly indistinguishable from the original."
            )

        explanation = (
            f"Truncated SVD reconstruction A_{k} = ∑_{{i=1}}^{{{k}}} σ_i u_i v_iᵀ. "
            f"By the Eckart-Young-Mirsky Theorem, A_{k} is the unique closest rank-{k} matrix to the original image "
            f"under both the Frobenius norm and spectral 2-norm. "
            f"{visual_desc}"
        )

        return ReconstructionResponse(
            k=k,
            total_components=max_k,
            retained_percentage=retained_pct,
            mode=req.mode,
            mse=round(mse, 4),
            psnr=round(psnr, 2),
            energy_preserved_percent=round(energy_percent, 2),
            compression_ratio=compression_ratio,
            original_data_url=original_url,
            reconstructed_data_url=reconstructed_url,
            difference_data_url=diff_url,
            explanation=explanation,
        )


reconstruction_service = ReconstructionService()

