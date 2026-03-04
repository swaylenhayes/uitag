"""Resolve paths to bundled benchmark assets."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

# Ordered: dark theme first, light theme second.
BENCHMARK_IMAGES = [
    "vscode-dark-1920x1080.png",
    "lmstudio-light-1920x1080.png",
]


def get_benchmark_image_paths() -> list[Path]:
    """Return filesystem paths to bundled benchmark images.

    Uses importlib.resources to locate assets inside the installed package.
    Context managers are entered and kept alive for the process lifetime —
    acceptable for a CLI tool that runs once and exits.

    Raises:
        FileNotFoundError: If a bundled image is missing from the package.
    """
    pkg = files("uitag.assets")
    paths: list[Path] = []
    for name in BENCHMARK_IMAGES:
        resource = pkg.joinpath(name)
        ctx = as_file(resource)
        path = ctx.__enter__()
        if not path.exists():
            msg = f"Bundled benchmark image not found: {name}"
            raise FileNotFoundError(msg)
        paths.append(path)
    return paths
