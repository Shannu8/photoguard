"""Image service for handling uploads, image caching, and format representations."""

import uuid
import os
import io
import json
import zipfile
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image

from app.utils.image_utils import load_image_from_bytes, array_to_base64_png
from app.utils.math_utils import rgb_to_grayscale
from app.models.schemas import ImageMetadataResponse, TensorMetadataResponse, TensorInfo


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

        # Phase 3 transformed cache
        self.transformed_rgb_array: Optional[np.ndarray] = None
        self.transformed_grayscale_array: Optional[np.ndarray] = None
        self.transformed_meta: Optional[Dict[str, Any]] = None

        # Phase 4 perturbation cache
        self.perturbed_rgb_array: Optional[np.ndarray] = None
        self.perturbed_grayscale_array: Optional[np.ndarray] = None
        self.perturbation_delta_array: Optional[np.ndarray] = None  # signed float64 (I' - I)
        self.perturbation_meta: Optional[Dict[str, Any]] = None
        self.intermediate_rgb_arrays: List[np.ndarray] = []
        self.individual_delta_arrays: List[np.ndarray] = []
        self.pipeline_steps_meta: List[Dict[str, Any]] = []
        self.perturbed_rgb_normalized: Optional[np.ndarray] = None
        self.perturbation_delta_normalized: Optional[np.ndarray] = None

    def set_transformed(self, rgb: np.ndarray, gray: np.ndarray, meta: Optional[Dict[str, Any]] = None):
        """Caches the latest transformed array without mutating original arrays."""
        self.transformed_rgb_array = rgb
        self.transformed_grayscale_array = gray
        self.transformed_meta = meta or {}

    def set_perturbed(
        self,
        rgb: np.ndarray,
        gray: np.ndarray,
        delta: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
        intermediate_rgbs: Optional[List[np.ndarray]] = None,
        individual_deltas: Optional[List[np.ndarray]] = None,
        pipeline_steps_meta: Optional[List[Dict[str, Any]]] = None,
        perturbed_rgb_normalized: Optional[np.ndarray] = None,
        perturbation_delta_normalized: Optional[np.ndarray] = None,
    ):
        """Caches the latest mathematically perturbed image and difference tensor without in-place mutation."""
        self.perturbed_rgb_array = rgb
        self.perturbed_grayscale_array = gray
        self.perturbation_delta_array = delta
        self.perturbation_meta = meta or {}
        self.intermediate_rgb_arrays = intermediate_rgbs or []
        self.individual_delta_arrays = individual_deltas or []
        self.pipeline_steps_meta = pipeline_steps_meta or []
        self.perturbed_rgb_normalized = perturbed_rgb_normalized
        self.perturbation_delta_normalized = perturbation_delta_normalized

    @property
    def id(self) -> str:
        return self.image_id

    @property
    def perturbation_metadata(self) -> Optional[Dict[str, Any]]:
        return self.perturbation_meta


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

    def get_tensor_metadata(self, image_id: str) -> TensorMetadataResponse:
        """Returns tensor shape, dtype, element counts and sizes for in-memory tensors."""
        img = self.get_image(image_id)
        has_trans = img.transformed_rgb_array is not None

        orig_rgb_info = TensorInfo(
            name="Original RGB Tensor",
            shape=list(img.rgb_array.shape),
            dtype=str(img.rgb_array.dtype),
            total_elements=int(img.rgb_array.size),
            estimated_size_bytes=int(img.rgb_array.nbytes),
        )
        orig_gray_info = TensorInfo(
            name="Original Grayscale Matrix",
            shape=list(img.grayscale_array.shape),
            dtype=str(img.grayscale_array.dtype),
            total_elements=int(img.grayscale_array.size),
            estimated_size_bytes=int(img.grayscale_array.nbytes),
        )

        trans_rgb_info = None
        trans_gray_info = None
        if has_trans:
            trans_rgb_info = TensorInfo(
                name="Transformed RGB Tensor",
                shape=list(img.transformed_rgb_array.shape),
                dtype=str(img.transformed_rgb_array.dtype),
                total_elements=int(img.transformed_rgb_array.size),
                estimated_size_bytes=int(img.transformed_rgb_array.nbytes),
            )
            trans_gray_info = TensorInfo(
                name="Transformed Grayscale Matrix",
                shape=list(img.transformed_grayscale_array.shape),
                dtype=str(img.transformed_grayscale_array.dtype),
                total_elements=int(img.transformed_grayscale_array.size),
                estimated_size_bytes=int(img.transformed_grayscale_array.nbytes),
            )

        has_pert = img.perturbed_rgb_array is not None
        pert_rgb_info = None
        pert_gray_info = None
        pert_delta_info = None
        if has_pert:
            pert_rgb_info = TensorInfo(
                name="Perturbed RGB Tensor",
                shape=list(img.perturbed_rgb_array.shape),
                dtype=str(img.perturbed_rgb_array.dtype),
                total_elements=int(img.perturbed_rgb_array.size),
                estimated_size_bytes=int(img.perturbed_rgb_array.nbytes),
            )
            if img.perturbed_grayscale_array is not None:
                pert_gray_info = TensorInfo(
                    name="Perturbed Grayscale Matrix",
                    shape=list(img.perturbed_grayscale_array.shape),
                    dtype=str(img.perturbed_grayscale_array.dtype),
                    total_elements=int(img.perturbed_grayscale_array.size),
                    estimated_size_bytes=int(img.perturbed_grayscale_array.nbytes),
                )
            if img.perturbation_delta_array is not None:
                pert_delta_info = TensorInfo(
                    name="Perturbation Delta Matrix",
                    shape=list(img.perturbation_delta_array.shape),
                    dtype=str(img.perturbation_delta_array.dtype),
                    total_elements=int(img.perturbation_delta_array.size),
                    estimated_size_bytes=int(img.perturbation_delta_array.nbytes),
                )

        return TensorMetadataResponse(
            has_transformed=has_trans,
            original_rgb=orig_rgb_info,
            original_grayscale=orig_gray_info,
            transformed_rgb=trans_rgb_info,
            transformed_grayscale=trans_gray_info,
            has_perturbed=has_pert,
            perturbed_rgb=pert_rgb_info,
            perturbed_grayscale=pert_gray_info,
            perturbation_delta=pert_delta_info,
        )

    def export_tensor(
        self, image_id: str, target: str, format_type: str
    ) -> Tuple[bytes, str, str]:
        """
        Exports complete numerical representation without modifying source arrays.
        target: 'original_rgb', 'transformed_rgb', 'original_grayscale', 'transformed_grayscale',
                'perturbed_rgb', 'perturbed_grayscale', 'perturbation', 'difference', 'absolute_difference', 'all'
        format_type: 'npy', 'npz', 'csv'
        Returns: (file_bytes, media_type, filename)
        """
        img = self.get_image(image_id)
        base_name = os.path.splitext(img.filename)[0]

        arrays_dict: Dict[str, np.ndarray] = {
            "original_rgb": img.rgb_array,
            "original_grayscale": img.grayscale_array,
        }
        if img.transformed_rgb_array is not None:
            arrays_dict["transformed_rgb"] = img.transformed_rgb_array
        if img.transformed_grayscale_array is not None:
            arrays_dict["transformed_grayscale"] = img.transformed_grayscale_array
        if img.perturbed_rgb_array is not None:
            arrays_dict["perturbed_rgb"] = img.perturbed_rgb_array
        if img.perturbed_grayscale_array is not None:
            arrays_dict["perturbed_grayscale"] = img.perturbed_grayscale_array
        if img.perturbation_delta_array is not None:
            arrays_dict["perturbation"] = img.perturbation_delta_array
            arrays_dict["difference"] = img.perturbation_delta_array
            arrays_dict["absolute_difference"] = np.abs(img.perturbation_delta_array)
        if img.perturbed_rgb_normalized is not None:
            arrays_dict["perturbed_rgb_normalized"] = img.perturbed_rgb_normalized
        if img.perturbation_delta_normalized is not None:
            arrays_dict["perturbation_normalized"] = img.perturbation_delta_normalized
            arrays_dict["delta_normalized"] = img.perturbation_delta_normalized

        # Include multi-perturbation pipeline intermediate tensors if available
        if img.intermediate_rgb_arrays:
            for idx, arr in enumerate(img.intermediate_rgb_arrays):
                arrays_dict[f"intermediate_step_{idx + 1}"] = arr
        if img.individual_delta_arrays:
            for idx, arr in enumerate(img.individual_delta_arrays):
                arrays_dict[f"individual_delta_{idx + 1}"] = arr

        format_clean = format_type.lower()

        # Handle JSON experiment metadata export
        if format_clean == "json" or target == "metadata":
            meta_payload = {
                "experiment": "MathOptic Perturbation Pipeline",
                "image_filename": img.filename,
                "image_shape": list(img.rgb_array.shape),
                "image_dtype": str(img.rgb_array.dtype),
                "pixel_range": [0, 255],
                "pipeline_metadata": img.perturbation_meta,
                "pipeline_steps": img.pipeline_steps_meta,
                "exported_arrays": list(arrays_dict.keys()),
                "binary_format_notice": (
                    "NumPy .npy and .npz files are binary numerical representations. "
                    "Do NOT open with a text editor. Load with Python: "
                    "`import numpy as np; data = np.load('filename.npz')`."
                ),
            }
            json_bytes = json.dumps(meta_payload, indent=2).encode("utf-8")
            return json_bytes, "application/json", f"{base_name}_experiment_metadata.json"

        if target != "all" and target not in arrays_dict:
            raise ValueError(
                f"Requested target '{target}' is not available. Please apply a transformation or perturbation first."
            )

        # Helper for CSV formatting
        def _get_csv_fmt(arr: np.ndarray) -> str:
            return "%d" if np.issubdtype(arr.dtype, np.integer) else "%.6f"

        # 1. NumPy NPY export
        if format_clean == "npy":
            if target == "all":
                bio_zip = io.BytesIO()
                with zipfile.ZipFile(bio_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for arr_name, arr_data in arrays_dict.items():
                        arr_bio = io.BytesIO()
                        np.save(arr_bio, arr_data)
                        zf.writestr(f"{base_name}_{arr_name}.npy", arr_bio.getvalue())
                bio_zip.seek(0)
                return bio_zip.getvalue(), "application/zip", f"{base_name}_all_tensors_npy.zip"
            else:
                arr = arrays_dict[target]
                bio = io.BytesIO()
                np.save(bio, arr)
                bio.seek(0)
                return bio.getvalue(), "application/octet-stream", f"{base_name}_{target}.npy"

        # 2. NumPy NPZ export
        elif format_clean == "npz":
            bio = io.BytesIO()
            if target == "all":
                np.savez(bio, **arrays_dict)
                bio.seek(0)
                return bio.getvalue(), "application/octet-stream", f"{base_name}_all_tensors.npz"
            else:
                np.savez(bio, **{target: arrays_dict[target]})
                bio.seek(0)
                return bio.getvalue(), "application/octet-stream", f"{base_name}_{target}.npz"

        # 3. CSV export
        elif format_clean == "csv":
            if target == "all":
                bio_zip = io.BytesIO()
                with zipfile.ZipFile(bio_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for arr_name, arr_data in arrays_dict.items():
                        fmt = _get_csv_fmt(arr_data)
                        if arr_data.ndim == 2:
                            sio = io.StringIO()
                            np.savetxt(sio, arr_data, fmt=fmt, delimiter=",")
                            zf.writestr(f"{base_name}_{arr_name}.csv", sio.getvalue())
                        elif arr_data.ndim == 3 and arr_data.shape[2] == 3:
                            for c_idx, c_name in enumerate(["red", "green", "blue"]):
                                sio = io.StringIO()
                                np.savetxt(sio, arr_data[:, :, c_idx], fmt=fmt, delimiter=",")
                                zf.writestr(f"{base_name}_{arr_name}_channel_{c_name}.csv", sio.getvalue())
                bio_zip.seek(0)
                return bio_zip.getvalue(), "application/zip", f"{base_name}_all_tensors_csv.zip"
            else:
                arr = arrays_dict[target]
                fmt = _get_csv_fmt(arr)
                if arr.ndim == 2:
                    sio = io.StringIO()
                    np.savetxt(sio, arr, fmt=fmt, delimiter=",")
                    return sio.getvalue().encode("utf-8"), "text/csv", f"{base_name}_{target}.csv"
                elif arr.ndim == 3 and arr.shape[2] == 3:
                    bio_zip = io.BytesIO()
                    with zipfile.ZipFile(bio_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for c_idx, c_name in enumerate(["red", "green", "blue"]):
                            sio = io.StringIO()
                            np.savetxt(sio, arr[:, :, c_idx], fmt=fmt, delimiter=",")
                            zf.writestr(f"{base_name}_{target}_channel_{c_name}.csv", sio.getvalue())
                    bio_zip.seek(0)
                    return bio_zip.getvalue(), "application/zip", f"{base_name}_{target}_rgb_csv.zip"
                else:
                    raise ValueError(f"Array shape {arr.shape} cannot be exported to CSV.")
        else:
            raise ValueError(f"Unsupported format '{format_type}'. Must be 'npy', 'npz', or 'csv'.")

    def get_image_png_bytes(
        self, image_id: str, target: str = "perturbed_rgb"
    ) -> Tuple[bytes, str, str]:
        """
        Exports the selected image as lossless 8-bit quantized PNG bytes.
        Does not modify stored internal tensors.
        target: 'perturbed_rgb', 'original_rgb', 'transformed_rgb', 'original_grayscale', 'perturbed_grayscale'
        Returns: (png_bytes, "image/png", filename)
        """
        img = self.get_image(image_id)
        base_name = os.path.splitext(img.filename)[0]
        target_clean = (target or "").strip().lower()

        if target_clean in ["perturbed_rgb", "perturbed"]:
            if img.perturbed_rgb_array is None:
                raise ValueError("No perturbation has been applied to this image yet.")
            arr = img.perturbed_rgb_array
            suffix = "processed"
        elif target_clean in ["transformed_rgb", "transformed"]:
            if img.transformed_rgb_array is None:
                raise ValueError("No transformation has been applied to this image yet.")
            arr = img.transformed_rgb_array
            suffix = "transformed"
        elif target_clean in ["original_rgb", "original"]:
            arr = img.rgb_array
            suffix = "original"
        elif target_clean in ["original_grayscale", "grayscale"]:
            arr = np.clip(np.round(img.grayscale_array), 0, 255).astype(np.uint8)
            suffix = "original_grayscale"
        elif target_clean in ["perturbed_grayscale"]:
            if img.perturbed_grayscale_array is None:
                raise ValueError("No perturbation has been applied to this image yet.")
            arr = np.clip(np.round(img.perturbed_grayscale_array), 0, 255).astype(np.uint8)
            suffix = "perturbed_grayscale"
        else:
            raise ValueError(
                f"Unsupported image download target: '{target}'. "
                "Supported targets: 'perturbed_rgb', 'transformed_rgb', 'original_rgb', 'original_grayscale', 'perturbed_grayscale'"
            )

        pil_img = Image.fromarray(arr)
        bio = io.BytesIO()
        pil_img.save(bio, format="PNG")
        bio.seek(0)
        return bio.getvalue(), "image/png", f"photoguard_{suffix}_{base_name}.png"



# Global singleton instance
image_service = ImageService()

