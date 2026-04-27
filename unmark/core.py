"""Core watermark-removal strategies."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

Corner = str  # "top-left" | "top-right" | "bottom-left" | "bottom-right"

CORNER_OPPOSITE: dict[Corner, Corner] = {
    "top-left": "top-right",
    "top-right": "top-left",
    "bottom-left": "bottom-right",
    "bottom-right": "bottom-left",
}


def _corner_slice(h: int, w: int, corner: Corner, size: int) -> tuple[slice, slice]:
    """Return (row_slice, col_slice) for the given corner of a (h, w) image."""
    if corner == "top-left":
        return slice(0, size), slice(0, size)
    if corner == "top-right":
        return slice(0, size), slice(w - size, w)
    if corner == "bottom-left":
        return slice(h - size, h), slice(0, size)
    if corner == "bottom-right":
        return slice(h - size, h), slice(w - size, w)
    raise ValueError(f"Unknown corner: {corner!r}")


def _flip_for_mirror(patch: np.ndarray, src_corner: Corner, dst_corner: Corner) -> np.ndarray:
    """Flip patch so the gradient direction aligns with destination corner."""
    src_v, src_h = src_corner.split("-")
    dst_v, dst_h = dst_corner.split("-")
    flip_h = src_h != dst_h
    flip_v = src_v != dst_v
    if flip_h and flip_v:
        return cv2.flip(patch, -1)
    if flip_h:
        return cv2.flip(patch, 1)
    if flip_v:
        return cv2.flip(patch, 0)
    return patch.copy()


def mirror_patch(
    img: np.ndarray,
    corner: Corner = "bottom-right",
    *,
    source_corner: Corner | None = None,
    patch_ratio: float = 0.22,
    full_radius_ratio: float = 0.42,
    fade_radius_ratio: float = 0.78,
) -> np.ndarray:
    """Replace a corner region with a feathered mirror of another corner.

    Best for watermarks on near-uniform backgrounds where opposite corners share
    the same vignette pattern (e.g., Gemini's bottom-right sparkle on indigo).

    Args:
        img: BGR image (uint8) as returned by ``cv2.imread``.
        corner: Where the watermark sits.
        source_corner: Where to copy clean pixels from. Defaults to the diagonally
            opposite corner.
        patch_ratio: Side length of the patch as a fraction of ``min(h, w)``.
        full_radius_ratio: Inside this fraction of the patch, replacement is full (1.0).
        fade_radius_ratio: Beyond this fraction, replacement is 0. Linear fade in between.
    """
    h, w = img.shape[:2]
    size = max(8, int(min(h, w) * patch_ratio))
    src_corner = source_corner or CORNER_OPPOSITE[corner]

    src_rows, src_cols = _corner_slice(h, w, src_corner, size)
    dst_rows, dst_cols = _corner_slice(h, w, corner, size)
    src_patch = _flip_for_mirror(img[src_rows, src_cols], src_corner, corner)

    # Build a feather mask anchored at the OUTER corner of the patch
    # (the corner closest to the image edge, which is where the watermark lives).
    outer_row, outer_col = _outer_anchor(corner, size)
    y_idx, x_idx = np.indices((size, size)).astype(np.float32)
    dist = np.sqrt((y_idx - outer_row) ** 2 + (x_idx - outer_col) ** 2)
    full_r = size * full_radius_ratio
    fade_r = size * fade_radius_ratio
    mask = np.where(
        dist <= full_r,
        1.0,
        np.where(dist < fade_r, (fade_r - dist) / (fade_r - full_r), 0.0),
    ).astype(np.float32)
    mask_3c = np.stack([mask] * 3, axis=2)

    target = img[dst_rows, dst_cols].astype(np.float32)
    blended = src_patch.astype(np.float32) * mask_3c + target * (1 - mask_3c)
    out = img.copy()
    out[dst_rows, dst_cols] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _outer_anchor(corner: Corner, size: int) -> tuple[int, int]:
    """Patch-local row/col of the corner that touches the image edge."""
    last = size - 1
    return {
        "top-left": (0, 0),
        "top-right": (0, last),
        "bottom-left": (last, 0),
        "bottom-right": (last, last),
    }[corner]


def inpaint_bright(
    img: np.ndarray,
    corner: Corner = "bottom-right",
    *,
    region_ratio: float = 0.15,
    brightness_threshold: int = 120,
    dilate_iters: int = 3,
    inpaint_radius: int = 10,
    flags: int = cv2.INPAINT_TELEA,
) -> np.ndarray:
    """Detect bright watermark pixels in a corner and inpaint them.

    Best for solid-color watermarks (white sparkles, light text) on darker backgrounds.
    """
    h, w = img.shape[:2]
    size = max(8, int(min(h, w) * region_ratio))
    rows, cols = _corner_slice(h, w, corner, size)

    region = img[rows, cols]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, region_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[rows, cols] = region_mask
    if dilate_iters > 0:
        kernel = np.ones((5, 5), np.uint8)
        full_mask = cv2.dilate(full_mask, kernel, iterations=dilate_iters)

    return cv2.inpaint(img, full_mask, inpaint_radius, flags)


Strategy = str  # "mirror" | "inpaint" | "hybrid"


def remove_watermark(
    img: np.ndarray | str | Path,
    *,
    strategy: Strategy = "mirror",
    corner: Corner = "bottom-right",
    **kwargs,
) -> np.ndarray:
    """Public entry point. Accepts an array or a path; returns a BGR uint8 array."""
    if isinstance(img, (str, Path)):
        loaded = cv2.imread(str(img))
        if loaded is None:
            raise FileNotFoundError(f"Could not read image: {img}")
        img = loaded

    if strategy == "mirror":
        return mirror_patch(img, corner=corner, **kwargs)
    if strategy == "inpaint":
        return inpaint_bright(img, corner=corner, **kwargs)
    if strategy == "hybrid":
        intermediate = mirror_patch(img, corner=corner, **kwargs.get("mirror", {}))
        return inpaint_bright(intermediate, corner=corner, **kwargs.get("inpaint", {}))
    raise ValueError(f"Unknown strategy: {strategy!r}")
