"""Conversion helpers for sending MineDojo RGB observations to a VLM."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def rgb_to_image(rgb: np.ndarray) -> Image.Image:
    """Convert MineDojo CHW or HWC RGB data into a Pillow image."""
    array = np.asarray(rgb)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D RGB array, got shape {array.shape}")
    if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected three color channels, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def encode_rgb_as_png_data_url(rgb: np.ndarray) -> Tuple[str, bytes]:
    """Return an OpenAI-compatible PNG data URL and the underlying bytes."""
    buffer = io.BytesIO()
    rgb_to_image(rgb).save(buffer, format="PNG", optimize=True)
    png_bytes = buffer.getvalue()
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}", png_bytes


def save_png(png_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)
