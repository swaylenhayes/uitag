"""Tests for benchmark CLI stats and formatting."""

import json

import pytest


def test_compute_stats_single_run():
    """Single run: stddev should be 0."""
    from uitag.bench_cli import compute_stats

    timings = [{"vision_ms": 100.0, "florence_total_ms": 200.0}]
    stats = compute_stats(timings)
    assert stats["vision_ms"]["mean"] == pytest.approx(100.0)
    assert stats["vision_ms"]["stddev"] == pytest.approx(0.0)
    assert stats["florence_total_ms"]["mean"] == pytest.approx(200.0)


def test_compute_stats_multiple_runs():
    """Multiple runs: correct mean and nonzero stddev."""
    from uitag.bench_cli import compute_stats

    timings = [
        {"vision_ms": 100.0, "florence_total_ms": 200.0},
        {"vision_ms": 110.0, "florence_total_ms": 210.0},
        {"vision_ms": 105.0, "florence_total_ms": 205.0},
    ]
    stats = compute_stats(timings)
    assert stats["vision_ms"]["mean"] == pytest.approx(105.0, abs=0.1)
    assert stats["vision_ms"]["stddev"] > 0


def test_compute_stats_missing_keys():
    """Keys present in some runs but not others get stats from available data."""
    from uitag.bench_cli import compute_stats

    timings = [
        {"vision_ms": 100.0, "tiling_ms": 2.0},
        {"vision_ms": 110.0},
    ]
    stats = compute_stats(timings)
    assert "vision_ms" in stats
    assert stats["vision_ms"]["mean"] == pytest.approx(105.0)
    assert stats["tiling_ms"]["mean"] == pytest.approx(2.0)


def test_format_table_renders_stages():
    """Table output contains stage names and timing values."""
    from uitag.bench_cli import compute_stats, format_table

    timings = [
        {
            "vision_ms": 977.0,
            "tiling_ms": 1.8,
            "florence_total_ms": 887.0,
            "merge_ms": 3.2,
            "annotate_ms": 42.0,
            "manifest_ms": 0.8,
        }
    ]
    stats = compute_stats(timings)
    machine = {"chip": "Apple M2 Max", "macos": "26.3", "uitag_version": "0.3.1"}
    table = format_table(
        stats=stats,
        machine_info=machine,
        image_name="test.png",
        image_size="1920x1080",
        runs=1,
        warmup=1,
        detection_count=151,
        ocr_mode="accurate",
    )
    assert "Vision" in table
    assert "Florence" in table
    assert "977" in table
    assert "151" in table
    assert "M2 Max" in table


def test_build_json_report_structure():
    """JSON report has required top-level keys."""
    from uitag.bench_cli import build_json_report, compute_stats

    timings = [{"vision_ms": 100.0, "florence_total_ms": 200.0}]
    stats = compute_stats(timings)
    machine = {"chip": "Apple M2 Max", "macos": "26.3", "uitag_version": "0.3.1"}
    report = build_json_report(
        stats=stats,
        machine_info=machine,
        image_name="test.png",
        image_size="1920x1080",
        runs=1,
        warmup=1,
        detection_count=151,
        ocr_mode="accurate",
    )
    data = json.loads(report)
    assert "machine" in data
    assert "stages" in data
    assert "summary" in data
    assert data["summary"]["detection_count"] == 151
    assert data["machine"]["chip"] == "Apple M2 Max"


def test_smart_dispatch_routes_benchmark(monkeypatch):
    """cli.main() routes 'benchmark' to bench_cli.benchmark_main."""
    import uitag.cli

    called_with = []

    def fake_benchmark_main(argv):
        called_with.append(argv)

    monkeypatch.setattr("uitag.bench_cli.benchmark_main", fake_benchmark_main)
    monkeypatch.setattr("sys.argv", ["uitag", "benchmark", "--runs", "1", "test.png"])

    uitag.cli.main()
    assert called_with == [["--runs", "1", "test.png"]]


def test_bundled_benchmark_images_exist():
    """Bundled benchmark images are accessible and are valid PNGs."""
    from uitag.assets.bundled import BENCHMARK_IMAGES, get_benchmark_image_paths

    paths = get_benchmark_image_paths()
    assert len(paths) == len(BENCHMARK_IMAGES)
    for path, expected_name in zip(paths, BENCHMARK_IMAGES):
        assert path.name == expected_name
        assert path.exists()
        # Verify PNG magic bytes
        with open(path, "rb") as f:
            header = f.read(4)
        assert header == b"\x89PNG"


def test_benchmark_main_nonexistent_image_exits():
    """benchmark_main with missing image exits with error."""
    from uitag.bench_cli import benchmark_main

    with pytest.raises(SystemExit):
        benchmark_main(["/nonexistent/image.png"])
