"""Image service for handling uploads, image caching, and format representations."""

import uuid
import os
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

from app.utils.image_utils import load_image_from_bytes, array_to_base64_png
from app.utils.math_utils import rgb_to_grayscale
from app.models.schemas import ImageMetadataResponse


class StoredImage:
    """Represents an uploaded image with its mathematical array representations."""
    def __init__(
        self,
        image_id: str,
        filename: str,
        raw_bytes: bytes,
        format_name: str,
        rgb_array: np.ndarray,
        grayscale_array: np.ndarray,
    ):
        self.image_id = image_id
        self.filename = filename
        self.raw_bytes = raw_bytes
        self.format_name = format_name
        self.rgb_array = rgb_array  # (H, W, 3) uint8
        self.grayscale_array = grayscale_array  # (H, W) float64
        self.height, self.width = rgb_array.shape[:2]
        self.channels = rgb_array.shape[2]
        self.total_pixels = self.height * self.width
        self.file_size_bytes = len(raw_bytes)

        # Precompute base64 previews
        self.rgb_data_url = array_to_base64_png(self.rgb_array, is_grayscale=False)
        self.grayscale_data_url = array_to_base64_png(self.grayscale_array, is_grayscale=True)


class ImageService:
    """Service managing images for mathematical analysis."""

    def __init__(self):
        # In-memory dictionary indexed by image_id
        self._store: Dict[str, StoredImage] = {}

    def upload_image(self, filename: str, raw_bytes: bytes) -> ImageMetadataResponse:
        """
        Processes and stores an uploaded image file.
        Validates format and never modifies original image bytes.
        """
        pil_img, rgb_array, detected_format = load_image_from_bytes(raw_bytes)
        grayscale_array = rgb_to_grayscale(rgb_array)

        image_id = str(uuid.uuid4())
        stored = StoredImage(
            image_id=image_id,
            filename=filename,
            raw_bytes=raw_bytes,
            format_name=detected_format,
            rgb_array=rgb_array,
            grayscale_array=grayscale_array,
        )
        self._store[image_id] = stored

        return self.get_metadata(image_id)

    def get_image(self, image_id: str) -> StoredImage:
        """Retrieves stored image or raises KeyError."""
        if image_id not in self._store:
            raise KeyError(f"Image with id '{image_id}' not found.")
        return self._store[image_id]

    def get_metadata(self, image_id: str) -> ImageMetadataResponse:
        """Returns metadata response for image."""
        img = self.get_image(image_id)
        return ImageMetadataResponse(
            image_id=img.image_id,
            filename=img.filename,
            width=img.width,
            height=img.height,
            format=img.format_name,
            channels=img.channels,
            total_pixels=img.total_pixels,
            color_mode="RGB",
            file_size_bytes=img.file_size_bytes,
            rgb_data_url=img.rgb_data_url,
            grayscale_data_url=img.grayscale_data_url,
        )


# Global singleton instance
image_service = ImageService()

