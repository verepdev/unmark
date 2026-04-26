"""Per-tool watermark presets."""

from __future__ import annotations

from typing import Callable

import numpy as np

from unmark.core import remove_watermark


def gemini(img: np.ndarray) -> np.ndarray:
    """Google Gemini stamps a bright diamond/sparkle in the bottom-right corner on every
    generated image. Inpaint detects the bright pixels and fills from immediate
    neighbors, so whatever content sits behind the sparkle (fog, gradient, sky,
    vignette) is preserved instead of being replaced with content from the opposite
    corner. Works on both uniform and non-uniform backgrounds."""
    return remove_watermark(
        img,
        strategy="inpaint",
        corner="bottom-right",
        region_ratio=0.20,
        brightness_threshold=200,
        dilate_iters=4,
        inpaint_radius=12,
    )


PRESETS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gemini": gemini,
}
