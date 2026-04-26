"""Smoke tests for unmark."""

import numpy as np
import pytest

from unmark.core import (
    CORNER_OPPOSITE,
    inpaint_bright,
    mirror_patch,
    remove_watermark,
)
from unmark.presets import PRESETS


def _indigo_image(size: int = 256) -> np.ndarray:
    """A flat indigo BGR image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = 220  # B
    img[..., 1] = 70  # G
    img[..., 2] = 60  # R
    return img


def _stamp_watermark(img: np.ndarray, corner: str = "bottom-right") -> np.ndarray:
    """Place a small bright square in the named corner to simulate a watermark."""
    out = img.copy()
    h, w = out.shape[:2]
    s = 20
    if corner == "bottom-right":
        out[h - s - 5 : h - 5, w - s - 5 : w - 5] = [255, 255, 255]
    elif corner == "top-left":
        out[5 : 5 + s, 5 : 5 + s] = [255, 255, 255]
    return out


def test_mirror_patch_removes_white_block():
    img = _stamp_watermark(_indigo_image())
    cleaned = mirror_patch(img, corner="bottom-right")
    h, w = cleaned.shape[:2]
    # The exact watermark pixels should now match background
    region = cleaned[h - 25 : h - 5, w - 25 : w - 5]
    assert region.mean(axis=(0, 1)).astype(int).tolist() != [255, 255, 255]
    # Should be close to the indigo background
    assert region[..., 0].mean() > 150  # blue channel high


def test_inpaint_bright_removes_white_block():
    img = _stamp_watermark(_indigo_image())
    cleaned = inpaint_bright(img, corner="bottom-right", brightness_threshold=200)
    h, w = cleaned.shape[:2]
    region = cleaned[h - 25 : h - 5, w - 25 : w - 5]
    assert region[..., 0].mean() > 150


def test_remove_watermark_dispatches_strategies():
    img = _stamp_watermark(_indigo_image())
    for strategy in ("mirror", "inpaint", "hybrid"):
        out = remove_watermark(img, strategy=strategy, corner="bottom-right")
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_unknown_strategy_raises():
    img = _indigo_image()
    with pytest.raises(ValueError):
        remove_watermark(img, strategy="not-a-strategy")


def test_corner_opposite_pairs_are_symmetric():
    for a, b in CORNER_OPPOSITE.items():
        assert CORNER_OPPOSITE[b] == a


def test_gemini_preset_is_callable():
    img = _stamp_watermark(_indigo_image())
    out = PRESETS["gemini"](img)
    assert out.shape == img.shape
