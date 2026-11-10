"""Image utilities for encoding, decoding, format handling, and heatmap visualization."""

import io
import base64
import numpy as np
from PIL import Image, ImageOps
import cv2
from typing import Tuple, Optional


ALLOWED_FORMATS = {"PNG", "JPEG", "JPG", "WEBP", "BMP"}


def array_to_base64_png(array: np.ndarray, is_grayscale: bool = False, max_preview_dim: Optional[int] = 1024) -> str:
    """
    Converts a NumPy uint8 or float array to a base64 encoded PNG data URL.
    Optionally scales down large display previews to prevent multi-megabyte JSON payload bloat,
    while leaving the underlying mathematical matrix completely intact.
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

    # For responsive browser display of large matrices, constrain preview data URL dimensions
    if max_preview_dim and max(img.width, img.height) > max_preview_dim:
        scale = max_preview_dim / max(img.width, img.height)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        preview_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    else:
        preview_img = img

    buffered = io.BytesIO()
    preview_img.save(buffered, format="PNG")
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
        diff = np.mean(np.abs(orig_f - recon_f), axis=2)
    else:
        diff = np.abs(orig_f - recon_f)

    # Normalize error to [0, 255] for colormap visualization
    max_diff = np.max(diff) if np.max(diff) > 0 else 1.0
    norm_diff = np.clip((diff / max_diff) * 255.0, 0, 255).astype(np.uint8)

    # Apply colormap (INFERNO gives high contrast for errors)
    colored = cv2.applyColorMap(norm_diff, cv2.COLORMAP_INFERNO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return array_to_base64_png(colored_rgb, is_grayscale=False)


def load_image_from_bytes(raw_bytes: bytes) -> Tuple[Image.Image, np.ndarray, str]:
    """
    Safely opens and validates an image from raw bytes.
    Handles EXIF orientation and decodes to a consistent 3-channel RGB uint8 array.
    Returns (PIL Image, RGB NumPy array, format).
    """
    if not raw_bytes or len(raw_bytes) == 0:
        raise ValueError("Uploaded image file is empty (0 bytes).")

    bio = io.BytesIO(raw_bytes)
    try:
        pil_img = Image.open(bio)
        detected_format = (pil_img.format or "").upper()
        if not detected_format:
            detected_format = "PNG"
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

    if detected_format not in ALLOWED_FORMATS:
        raise ValueError(
            f"Unsupported format '{detected_format}'. Supported formats: {', '.join(sorted(ALLOWED_FORMATS))}"
        )

    try:
        # Correct rotation based on EXIF tag if present
        pil_img = ImageOps.exif_transpose(pil_img)
        # Force pixel buffer loading to catch truncated or corrupted streams early
        pil_img.load()
        # Convert to RGB (strips alpha channel safely while keeping pure color data)
        rgb_img = pil_img.convert("RGB")
        rgb_array = np.array(rgb_img, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Image decoding failed: {str(e)}")

    if rgb_array.size == 0 or rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("Decoded image tensor is empty or invalid.")

    return pil_img, rgb_array, detected_format
