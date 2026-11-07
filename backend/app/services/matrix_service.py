"""Matrix service for pixel inspection, channel statistics, and rank calculation."""

import numpy as np
from typing import Dict, Any, Tuple
from app.services.image_service import StoredImage
from app.utils.math_utils import (
    format_pixel_calculation,
    compute_channel_stats,
    compute_numerical_rank,
)
from app.models.schemas import (
    PixelInspectionResponse,
    ImageStatisticsResponse,
    ChannelStats,
    MatrixRankResponse,
)


class MatrixService:
    """Service performing mathematical matrix analysis on images."""

    @staticmethod
    def inspect_pixel(
        stored: StoredImage, x: int, y: int, radius: int = 3
    ) -> PixelInspectionResponse:
        """
        Inspects pixel at (x, y) (where x is col index 0..W-1, y is row index 0..H-1).
        Provides exact RGB values, transparent grayscale calculation, and local submatrix slice.
        """
        w, h = stored.width, stored.height
        # Clamp coordinates safely
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        r_val = int(stored.rgb_array[y, x, 0])
        g_val = int(stored.rgb_array[y, x, 1])
        b_val = int(stored.rgb_array[y, x, 2])

        y_exact, calc_str = format_pixel_calculation(r_val, g_val, b_val)
        y_uint8 = int(np.clip(round(y_exact), 0, 255))

        # Extract local submatrix around (x, y)
        r_start = max(0, y - radius)
        r_end = min(h, y + radius + 1)
        c_start = max(0, x - radius)
        c_end = min(w, x + radius + 1)

        sub_r = stored.rgb_array[r_start:r_end, c_start:c_end, 0].tolist()
        sub_g = stored.rgb_array[r_start:r_end, c_start:c_end, 1].tolist()
        sub_b = stored.rgb_array[r_start:r_end, c_start:c_end, 2].tolist()
        sub_gray = np.round(stored.grayscale_array[r_start:r_end, c_start:c_end], 2).tolist()

        return PixelInspectionResponse(
            x=x,
            y=y,
            r=r_val,
            g=g_val,
            b=b_val,
            grayscale_exact=round(y_exact, 4),
            grayscale_uint8=y_uint8,
            calculation_steps=calc_str,
            submatrix_center=(x, y),
            submatrix_bounds={
                "row_start": r_start,
                "row_end": r_end - 1,
                "col_start": c_start,
                "col_end": c_end - 1,
            },
            submatrix_r=sub_r,
            submatrix_g=sub_g,
            submatrix_b=sub_b,
            submatrix_grayscale=sub_gray,
        )

    @staticmethod
    def get_image_statistics(stored: StoredImage) -> ImageStatisticsResponse:
        """Calculates descriptive statistics and 256-bin histograms for all channels."""
        gray_stats = compute_channel_stats(stored.grayscale_array)
        r_stats = compute_channel_stats(stored.rgb_array[:, :, 0])
        g_stats = compute_channel_stats(stored.rgb_array[:, :, 1])
        b_stats = compute_channel_stats(stored.rgb_array[:, :, 2])

        return ImageStatisticsResponse(
            grayscale=ChannelStats(**gray_stats),
            red=ChannelStats(**r_stats),
            green=ChannelStats(**g_stats),
            blue=ChannelStats(**b_stats),
        )

    @staticmethod
    def analyze_matrix_rank(stored: StoredImage) -> MatrixRankResponse:
        """
        Calculates mathematical rank of the grayscale image matrix using SVD,
        identifying whether the matrix is full rank and explaining linear independence.
        """
        gray = stored.grayscale_array
        rows, cols = gray.shape
        max_possible = min(rows, cols)

        rank, tol, is_full, singular_vals = compute_numerical_rank(gray)

        # Top 10 and bottom 10 singular values for educational inspection
        top_s = [round(float(s), 4) for s in singular_vals[:10]]
        bottom_s = [round(float(s), 6) for s in singular_vals[-10:]]

        explanation = (
            f"The image matrix has dimensions {rows} × {cols}. "
            f"The computed numerical rank is {rank} out of a maximum possible rank of {max_possible}. "
            f"{'The matrix is mathematically full rank, meaning all ' + str(max_possible) + ' rows/columns are linearly independent.' if is_full else 'The matrix is rank-deficient by ' + str(max_possible - rank) + ' dimensions, indicating linear dependency among rows or columns.'} "
            f"Numerical rank is evaluated using SVD tolerance threshold tol = {tol:.4e}."
        )

        scientific_notes = (
            "CRITICAL MATHEMATICAL DISTINCTION: "
            "Matrix rank is a global algebraic property representing the dimension of the vector space "
            "spanned by the image rows or columns. It measures global linear independence across scanlines. "
            "Matrix rank DOES NOT measure or identify individual important image pixels or local visual saliency. "
            "In natural photographic images, subtle noise and varied textures usually make the image matrix numerically full rank, "
            "even though the majority of mathematical energy is concentrated in a tiny fraction of principal singular components."
        )

        return MatrixRankResponse(
            rows=rows,
            cols=cols,
            rank=rank,
            is_full_rank=is_full,
            numerical_tolerance=tol,
            max_possible_rank=max_possible,
            singular_values_top=top_s,
            singular_values_bottom=bottom_s,
            explanation=explanation,
            scientific_notes=scientific_notes,
        )


matrix_service = MatrixService()

