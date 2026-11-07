"""Image utilities for encoding, decoding, format handling, and heatmap visualization."""

import io
import base64
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional


ALLOWED_FORMATS = {"PNG", "JPEG", "JPG", "WEBP", "BMP"}


def array_to_base64_png(array: np.ndarray, is_grayscale: bool = False) -> str:
    """
    Converts a NumPy uint8 or float array to a base64 encoded PNG data URL.
    """
    clipped = np.clip(array, 0, 255).astype(np.uint8)

    if is_grayscale or clipped.ndim == 2:
        img = Image.fromarray(clipped, mode='L')
    elif clipped.ndim == 3 and clipped.shape[2] == 3:
        img = Image.fromarray(clipped, mode='RGB')
    elif clipped.ndim == 3 and clipped.shape[2] == 4:
        img = Image.fromarray(clipped, mode='RGBA')
    else:
        raise ValueError(f"Unsupported array shape for PNG conversion: {array.shape}")

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def generate_error_heatmap_base64(original: np.ndarray, reconstructed: np.ndarray) -> str:
    """
    Generates a visual heatmap of absolute pixel error |original - reconstructed|
    using the OpenCV INFERNO colormap.
    """
    orig_f = original.astype(np.float64)
    recon_f = reconstructed.astype(np.float64)

    if orig_f.ndim == 3:
        # For RGB, compute Euclidean or mean absolute error per pixel
        diff = np.mean(np.abs(orig_f - recon_f), axis=2)
    else:
        diff = np.abs(orig_f - recon_f)

    # Normalize error to [0, 255] for colormap visualization
    max_diff = np.max(diff) if np.max(diff) > 0 else 1.0
    norm_diff = np.clip((diff / max_diff) * 255.0, 0, 255).astype(np.uint8)

    # Apply colormap (INFERNO gives high contrast for errors)
    colored = cv2.applyColorMap(norm_diff, cv2.COLORMAP_INFERNO)
    # OpenCV uses BGR by default, convert to RGB for PIL
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return array_to_base64_png(colored_rgb, is_grayscale=False)


def load_image_from_bytes(raw_bytes: bytes) -> Tuple[Image.Image, np.ndarray, str]:
    """
    Safely opens and validates an image from raw bytes.
    Returns (PIL Image, RGB NumPy array, format).
    """
    bio = io.BytesIO(raw_bytes)
    try:
        pil_img = Image.open(bio)
        detected_format = (pil_img.format or "PNG").upper()
    except Exception as e:
        raise ValueError(f"Could not open image file: {str(e)}")

    if detected_format not in ALLOWED_FORMATS:
        raise ValueError(
            f"Unsupported format '{detected_format}'. Supported formats: {', '.join(sorted(ALLOWED_FORMATS))}"
        )

    # Convert to RGB to ensure a consistent 3-channel mathematical matrix representation
    # (strips alpha channel safely while keeping pure color data)
    rgb_img = pil_img.convert("RGB")
    rgb_array = np.array(rgb_img, dtype=np.uint8)

    return pil_img, rgb_array, detected_format

