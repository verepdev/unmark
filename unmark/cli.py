"""Command-line interface for unmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from unmark import __version__
from unmark.core import remove_watermark
from unmark.presets import PRESETS


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unmark",
        description="Strip Google Gemini's watermark from generated images. Fast, local, no API.",
    )
    p.add_argument("inputs", nargs="+", type=Path, help="Input image(s)")
    p.add_argument("-o", "--output", type=Path, help="Output file (single) or directory (batch)")
    p.add_argument(
        "--batch", action="store_true", help="Treat inputs as batch; --output must be a directory"
    )
    p.add_argument(
        "--tool",
        choices=sorted(PRESETS.keys()),
        help="Use a per-tool preset (overrides --strategy and --corner)",
    )
    p.add_argument(
        "--strategy",
        choices=["mirror", "inpaint", "hybrid"],
        default="mirror",
        help="Removal strategy when no --tool preset is given (default: mirror)",
    )
    p.add_argument(
        "--corner",
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        default="bottom-right",
        help="Corner where the watermark sits (default: bottom-right)",
    )
    p.add_argument(
        "--source", help="Source corner for mirror strategy (default: diagonally opposite)"
    )
    p.add_argument(
        "--threshold", type=int, default=120, help="Brightness threshold for inpaint strategy"
    )
    p.add_argument("-v", "--version", action="version", version=f"unmark {__version__}")
    return p


def _process_one(src: Path, dst: Path, args: argparse.Namespace) -> None:
    img = cv2.imread(str(src))
    if img is None:
        print(f"unmark: cannot read {src}", file=sys.stderr)
        sys.exit(1)

    if args.tool:
        result = PRESETS[args.tool](img)
    elif args.strategy == "mirror":
        kwargs = {}
        if args.source:
            kwargs["source_corner"] = args.source
        result = remove_watermark(img, strategy="mirror", corner=args.corner, **kwargs)
    elif args.strategy == "inpaint":
        result = remove_watermark(
            img, strategy="inpaint", corner=args.corner, brightness_threshold=args.threshold
        )
    else:  # hybrid
        result = remove_watermark(img, strategy="hybrid", corner=args.corner)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), result)
    print(f"unmark: {src} -> {dst}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.batch:
        if not args.output or args.output.suffix:
            print("unmark: --batch requires --output to be a directory", file=sys.stderr)
            return 2
        for src in args.inputs:
            dst = args.output / f"{src.stem}.clean{src.suffix}"
            _process_one(src, dst, args)
        return 0

    if len(args.inputs) != 1:
        print("unmark: pass --batch with multiple inputs, or a single input", file=sys.stderr)
        return 2
    src = args.inputs[0]
    dst = args.output or src.with_name(f"{src.stem}.clean{src.suffix}")
    _process_one(src, dst, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
