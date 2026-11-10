"""Tests for image upload validation across all supported formats, edge cases, and limits."""

import asyncio
import io
import pytest
from PIL import Image
from fastapi import HTTPException
from app.api.routes import upload_image
from app.main import health_check
from app.services.image_service import image_service


class DummyUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str = "image/png"):
        self.filename = filename
        self._content = content
        self.content_type = content_type

    async def read(self):
        return self._content


def test_supported_formats_upload():
    """Verify PNG, JPEG, WEBP, and BMP format uploads succeed and preserve metadata."""
    async def _test():
        formats = ["PNG", "JPEG", "WEBP", "BMP"]

        for fmt in formats:
            buf = io.BytesIO()
            img = Image.new("RGB", (80, 60), color=(120, 180, 240))
            img.save(buf, format=fmt)
            raw_bytes = buf.getvalue()

            file_obj = DummyUploadFile(f"sample.{fmt.lower()}", raw_bytes, f"image/{fmt.lower()}")
            res = await upload_image(file_obj)

            assert res.image_id is not None
            assert res.width == 80
            assert res.height == 60
            assert res.channels == 3
            assert res.total_pixels == 4800
            assert res.format == fmt
            assert res.rgb_data_url.startswith("data:image/png;base64,")
            assert res.grayscale_data_url.startswith("data:image/png;base64,")

            # Verify underlying stored image array
            stored = image_service.get_image(res.image_id)
            assert stored.rgb_array.shape == (60, 80, 3)
            assert stored.grayscale_array.shape == (60, 80)
            assert len(stored.raw_bytes) == len(raw_bytes)

    asyncio.run(_test())


def test_empty_file_upload_rejected():
    """Verify 0-byte upload is rejected with HTTP 400 and clear message."""
    async def _test():
        file_obj = DummyUploadFile("empty.png", b"")
        with pytest.raises(HTTPException) as exc_info:
            await upload_image(file_obj)

        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail.lower()

    asyncio.run(_test())


def test_invalid_corrupted_file_upload_rejected():
    """Verify corrupted non-image bytes are rejected with HTTP 400."""
    async def _test():
        fake_data = b"This is not a valid image byte stream."
        file_obj = DummyUploadFile("corrupt.png", fake_data)

        with pytest.raises(HTTPException) as exc_info:
            await upload_image(file_obj)

        assert exc_info.value.status_code == 400
        assert "invalid image file" in exc_info.value.detail.lower()

    asyncio.run(_test())


def test_unsupported_format_rejected():
    """Verify unsupported formats (e.g. TIFF, GIF, PDF) are rejected with HTTP 400."""
    async def _test():
        buf = io.BytesIO()
        img = Image.new("RGB", (40, 40))
        img.save(buf, format="GIF")
        gif_bytes = buf.getvalue()

        file_obj = DummyUploadFile("animation.gif", gif_bytes, "image/gif")
        with pytest.raises(HTTPException) as exc_info:
            await upload_image(file_obj)

        assert exc_info.value.status_code == 400
        assert "unsupported" in exc_info.value.detail.lower()

    asyncio.run(_test())


def test_oversized_file_upload_rejected():
    """Verify file exceeding 20MB limit is rejected with HTTP 413."""
    async def _test():
        # 21 MB dummy payload
        oversized_data = b"\x00" * (21 * 1024 * 1024)
        file_obj = DummyUploadFile("huge.png", oversized_data)

        with pytest.raises(HTTPException) as exc_info:
            await upload_image(file_obj)

        assert exc_info.value.status_code == 413
        assert "20mb" in exc_info.value.detail.lower()

    asyncio.run(_test())


def test_health_check_endpoint():
    """Verify health check endpoint returns online status."""
    res = health_check()
    assert res["status"] == "online"
    assert "Phase 1" in res["phases_active"][0]

