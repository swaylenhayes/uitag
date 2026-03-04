# uitag — Roadmap

_Last updated: 2026-03-03_
_Status: Sprints 1-3 complete. Post-launch features (batch, benchmark) shipped in v0.3.1._

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Detection pipeline | Stable | 1920x1080 → 151 detections in ~1s (fast OCR + optimized MLX) |
| README | Up to date | Architecture, quick start, performance, research background |
| CLI | Working | `uitag`, `uitag batch`, `uitag benchmark` |
| MIT license | Done | No AGPL contamination |
| Test suite | 87 tests | Fast/slow split, all passing |
| pyproject.toml | Complete | Metadata, deps, entry points, optional groups |
| Backend abstraction | Complete | MLX default + CoreML option, `--backend` flag |
| JSON manifest | Stable | Structured output consumed by downstream tools |
| Research docs | Documented | Object-aware tiling, Florence-2 tiling discovery, 14-model benchmark |

---

## Completed Work

### Sprint 1: CI + Documentation (2026-02-27)

- [x] **S1.1:** GitHub Actions CI — `.github/workflows/test.yml`, Python 3.10/3.11/3.12, badge in README
- [x] **S1.2:** Example output — hero image + redacted manifest in `docs/examples/`
- [x] **S1.3:** README refresh — perf numbers, CI badge, hero image, backend docs
- [x] **S1.4:** Pre-commit config — ruff check + format, codebase normalized

### Sprint 2: Distribution (2026-02-27)

- [x] **S2.1:** PyPI publish — v0.2.2 live on PyPI + TestPyPI, trusted publishers (OIDC), tag-triggered workflow
- [x] **S2.2:** GitHub Release — [v0.2.0](https://github.com/swaylenhayes/uitag/releases) published
- [x] **S2.3:** Issue templates — bug report + feature request (YAML forms)

### Sprint 3: Contributor Experience (2026-02-27)

- [x] **S3.1:** CONTRIBUTING.md — setup, architecture overview, PR workflow
- [x] **S3.2:** Integration examples — `examples/use_as_library.py` + `examples/custom_backend.py`
- [x] **S3.3:** Manifest JSON Schema — `uitag/schema.json` (Draft 2020-12) + 5 validation tests

### Post-Sprint Fixes (2026-02-27)

- [x] Pip install fix — Swift source bundled in package (`uitag/tools/`)
- [x] `run_pipeline` exported from `__init__.py`
- [x] README Quick Start leads with `pip install uitag`
- [x] Apache-2.0 SPDX headers → MIT (5 files)
- [x] Stale perf numbers updated in `docs/research.md`

### v0.3.x Features

- [x] **Batch CLI** (`uitag batch <dir>`) — Process folders of screenshots in one command (v0.3.1)
- [x] **Benchmark CLI** (`uitag benchmark`) — Per-stage timing with stats across N runs, bundled reference images (v0.3.1)
- [x] **Per-stage timing instrumentation** — All 6 pipeline stages timed in manifest output (v0.3.1)
- [x] **API reference docs** — `docs/api.md` with functions, types, manifest schema
- [x] **Performance docs** — `docs/performance.md` with stage breakdown, backend comparison
- [x] **VHS CLI demo GIF** — Animated terminal recording embedded in README Quick Start
- [x] **Bundled benchmark images** — Dark (VS Code) + light (LM Studio) reference screenshots, `uitag benchmark` runs both by default

---

## Up Next

- [ ] **Pipeline architecture visual** — Replace ASCII diagram in README with SVG/image

---

## Parked

| Item | Why parked | Resume trigger |
|------|-----------|----------------|
| CoreML as AUTO default | MLX is faster on idle GPU | Profiling shows otherwise |
| GPU load detection in selector | Not needed for single-user CLI | Multi-process use cases |
| Florence-2 task token exploration | Current task tokens work well | Quality issues surface |
| Parallel quadrant inference | Sequential is already ~650ms | User demand |
| Additional model support | Scope creep risk | Community requests |

---

_This is a living document. Update as the project evolves._
