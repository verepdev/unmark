"""unmark — Strip AI-generated image watermarks."""

from unmark.core import remove_watermark
from unmark.presets import PRESETS

__version__ = "0.1.0"
__all__ = ["remove_watermark", "PRESETS"]
