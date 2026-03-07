"""uitag batch — CLI subcommand for batch image processing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def discover_images(directory: Path) -> list[Path]:
    """Find all supported image files in a directory (non-recursive, sorted).

    Args:
        directory: Path to scan for images.

    Returns:
        Sorted list of image file paths.
    """
    images = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda p: p.name.lower())


def format_progress(
    *,
    index: int,
    total: int,
    name: str,
    elapsed_s: float | None = None,
    count: int | None = None,
    error: str | None = None,
) -> str:
    """Format a single progress line.

    Args:
        index: Zero-based image index.
        total: Total number of images.
        name: Image filename.
        elapsed_s: Processing time in seconds (for success).
        count: Detection count (for success).
        error: Error message (for failure).
    """
    prefix = f"[{index + 1}/{total}]"
    dots = "." * max(1, 40 - len(name))
    if error is not None:
        return f"{prefix} {name} {dots} FAILED ({error})"
    return f"{prefix} {name} {dots} {elapsed_s:.1f}s  {count} elements"


def format_summary(
    *,
    succeeded: int,
    failed: int,
    total_detections: int,
    total_seconds: float,
    output_dir: str,
    is_tty: bool = False,
) -> str:
    """Format the batch completion summary."""
    done_line = f"Done: {total_detections} detections in {total_seconds:.1f}s"
    if failed > 0:
        done_line += f" ({failed} failed)"
    if is_tty:
        done_line = f"\033[1;32m{done_line}\033[0m"
    return (
        f"\n{done_line}"
        f"\nOutput: {succeeded} images, {succeeded} manifests in {Path(output_dir).resolve()}/"
    )


def batch_main(argv: list[str] | None = None) -> None:
    """Entry point for ``uitag batch``."""
    parser = argparse.ArgumentParser(
        prog="uitag batch",
        description="Batch process images through the uitag detection pipeline",
    )
    parser.add_argument(
        "path",
        nargs="+",
        help="Directory containing images, or individual image paths",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="uitag-output",
        help="Output directory (default: uitag-output/)",
    )
    parser.add_argument("--fast", action="store_true", help="Use fast OCR mode")
    parser.add_argument(
        "--backend",
        choices=["auto", "coreml", "mlx"],
        default="auto",
        help="Detection backend",
    )

    args = parser.parse_args(argv)

    # Resolve image list
    image_paths: list[Path] = []
    for p_str in args.path:
        p = Path(p_str)
        if p.is_dir():
            found = discover_images(p)
            if not found:
                print(f"No images found in {p}", file=sys.stderr)
                sys.exit(1)
            image_paths.extend(found)
        elif p.is_file():
            image_paths.append(p)
        else:
            print(f"Error: Path not found: {p}", file=sys.stderr)
            sys.exit(1)

    if not image_paths:
        print("No images to process.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr_mode = "fast" if args.fast else "fine"
    ocr_recognition = "fast" if args.fast else "accurate"

    # Load backend once
    from uitag.backends.selector import BackendPreference, select_backend

    preference = BackendPreference(args.backend)
    backend = select_backend(preference=preference)

    # Warm the backend import before starting timer
    info = backend.info()
    source_label = args.path[0].rstrip("/") if len(args.path) == 1 else f"{len(args.path)} paths"
    print(f"Running pipeline on: {len(image_paths)} images in {source_label}/")
    print(f"Backend: {info.name} ({info.device}) | OCR mode: {ocr_mode}\n")

    # Process
    from uitag.run import run_pipeline

    succeeded = 0
    failed = 0
    total_detections = 0
    t_total = time.perf_counter()

    for i, img_path in enumerate(image_paths):
        try:
            t0 = time.perf_counter()
            result, annotated, manifest = run_pipeline(
                str(img_path),
                recognition_level=ocr_recognition,
                backend=backend,
            )
            elapsed = time.perf_counter() - t0

            # Save outputs
            stem = img_path.stem
            annotated.save(out_dir / f"{stem}-uitag.png")
            (out_dir / f"{stem}-uitag-manifest.json").write_text(manifest)

            det_count = len(result.detections)
            total_detections += det_count
            print(
                format_progress(
                    index=i,
                    total=len(image_paths),
                    name=img_path.name,
                    elapsed_s=elapsed,
                    count=det_count,
                )
            )
            succeeded += 1

        except Exception as exc:
            print(
                format_progress(
                    index=i,
                    total=len(image_paths),
                    name=img_path.name,
                    error=str(exc),
                )
            )
            failed += 1

    total_time = time.perf_counter() - t_total
    print(
        format_summary(
            succeeded=succeeded,
            failed=failed,
            total_detections=total_detections,
            total_seconds=total_time,
            output_dir=str(out_dir),
            is_tty=sys.stdout.isatty(),
        )
    )
