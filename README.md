# unmark

> Strip Google Gemini's watermark from generated images. Fast, local, no API.

`unmark` is a command-line tool that removes the small sparkle Google Gemini stamps in the bottom-right corner of every image it generates. It works fully offline using deterministic image-processing strategies — no cloud calls, no API keys, no telemetry.

## Why

Gemini stamps a small watermark on every image it generates. Generic inpainting tools either leave visible artifacts, demand manual masking, or send your image to a remote server. `unmark` knows exactly where the Gemini sparkle sits and removes it cleanly with image symmetry + content-aware fill — locally, in seconds.

## Install

```bash
pip install unmark
```

Or from source:

```bash
git clone https://github.com/verepdev/unmark.git
cd unmark
pip install -e .
```

Requires Python 3.10+.

## Usage

### Single image

```bash
unmark input.png -o output.png
```

### Tool-specific preset

```bash
unmark gemini-output.png -o clean.png --tool gemini
```

### Batch

```bash
unmark *.png --batch -o cleaned/
```

### Custom strategy

```bash
# Mirror-patch from another corner
unmark img.png -o out.png --strategy mirror --corner bottom-right --source bottom-left

# Targeted inpaint
unmark img.png -o out.png --strategy inpaint --threshold 120
```

## How it works

`unmark` ships three strategies, picked automatically based on the watermark's properties:

| Strategy | Best for | How it works |
|---|---|---|
| `mirror` | Watermarks in a corner of a near-uniform background (Gemini sparkle) | Copies the symmetric patch from the opposite corner, mirrors it, blends with feathered edges |
| `inpaint` | Watermarks with detectable color contrast | Builds a brightness/color mask, runs OpenCV TELEA inpainting on detected pixels |
| `hybrid` | Complex backgrounds | Mirror-patch first, then targeted inpaint on residual artifacts |

Tool presets bundle the right strategy + parameters for known watermark patterns.

## What it handles

| Mode | What it does |
|---|---|
| `--tool gemini` | Removes the bottom-right sparkle/diamond Gemini stamps on every generated image |
| Generic flags | `--strategy mirror|inpaint|hybrid` + `--corner` for any other corner-watermark on a near-uniform background |

## Legal & ethics

This tool is intended for **personal use** with images you generated yourself. Removing watermarks from images you do not own may violate the source tool's terms of service. **You are responsible for how you use this tool.**

## Contributing

Issues and PRs welcome, especially:
- Better strategies for non-uniform backgrounds
- Performance improvements
- Tests on edge cases (small images, very dense backgrounds, etc.)
- A preset for another AI tool you actually use — open an issue first so we can discuss the approach before you write code

Dev setup:
```bash
git clone https://github.com/verepdev/unmark.git
cd unmark
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pre-commit install   # enable format + lint hook on every commit
pytest
```

Code style: Ruff for lint, type hints on public functions, short focused functions.

## License

[MIT](LICENSE) — by [@verepdev](https://github.com/verepdev).
