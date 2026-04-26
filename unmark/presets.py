"""Per-tool watermark presets."""

from __future__ import annotations

from typing import Callable

import numpy as np

from unmark.core import remove_watermark


def gemini(img: np.ndarray) -> np.ndarray:
    """Google Gemini puts a small light sparkle in the bottom-right corner on the
    generated image's background. Mirror-patch from bottom-left works cleanly when
    the background is near-uniform (typical for portrait avatars)."""
    return remove_watermark(
        img,
        strategy="mirror",
        corner="bottom-right",
        patch_ratio=0.22,
        full_radius_ratio=0.42,
        fade_radius_ratio=0.78,
    )


PRESETS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gemini": gemini,
}
