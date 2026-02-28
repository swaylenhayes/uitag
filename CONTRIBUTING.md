# Contributing to SomFlow

## Quick Start

```bash
# Clone and install
git clone https://github.com/swaylenhayes/somflow.git
cd somflow
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Set up pre-commit (optional but recommended)
pip install pre-commit
pre-commit install
```

## Architecture

SomFlow runs a 6-stage detection pipeline:

```
Screenshot → [1] Apple Vision → [2] Tiling → [3] Florence-2 → [4] Merge → [5] Annotate → [6] Manifest
```

| Module | What it does |
|--------|-------------|
| `somflow/vision.py` | Apple Vision via Swift subprocess (text + rectangles) |
| `somflow/quadrants.py` | Object-aware image tiling (avoids splitting UI elements) |
| `somflow/florence.py` | Florence-2 detection token parsing |
| `somflow/merge.py` | IoU-based deduplication with source priority |
| `somflow/annotate.py` | SoM numbered overlay rendering |
| `somflow/manifest.py` | JSON manifest generation |
| `somflow/run.py` | Pipeline orchestrator — `run_pipeline()` is the main entry point |
| `somflow/backends/` | `DetectionBackend` protocol + MLX/CoreML implementations |

## Making Changes

1. Create a branch: `git checkout -b feat/your-feature` or `fix/your-fix`
2. Write tests first when possible
3. Run the linter: `uv run ruff check .`
4. Run tests: `uv run pytest`
5. Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
6. Open a PR against `main`

### Test Split

- `uv run pytest` — fast tests only (~50 tests, no model required)
- `uv run pytest --run-slow` — includes tests that load Florence-2 (requires macOS + model download)

Tests marked `@pytest.mark.slow` need the Florence-2 model and a macOS system with Apple Vision. CI runs fast tests only.

## What's Welcome

- **Bug fixes** — especially around edge cases in detection merging or tiling
- **New backends** — implement `DetectionBackend` protocol (see `somflow/backends/base.py`)
- **Test coverage** — more edge cases for quadrant splitting, IoU merging
- **Documentation** — examples, tutorials, manifest format docs

See [open issues](https://github.com/swaylenhayes/somflow/issues) for specific ideas.
