# uitag — Flywheel Launch Roadmap

_Last updated: 2026-03-02_
_Status: Sprints 1-3 complete. Post-launch features (batch, benchmark) shipped in v0.3.1._

---

## The Flywheel

```
Install & try it → See it work → Tell someone / use in a project
       ↑                                        |
       |                                        v
  Improved by feedback  ←←←  Issues, PRs, ideas
```

**Goal:** Get uitag installed, used, and talked about. Every friction point in the install → try → share loop slows the flywheel. This plan removes those friction points.

---

## What We Have (Strengths)

These are real and defensible — not aspirational:

| Asset | Status | Notes |
|-------|--------|-------|
| Working pipeline | Shipping | 1920x1080 → 151 detections in ~1s (fast OCR + optimized MLX) |
| Clean README | Good | Problem statement, architecture, quick start, research background |
| CLI entry point | Working | `uitag screenshot.png` works after install |
| MIT license | Done | Clean IP, no AGPL contamination |
| Test suite | 87 tests | Fast/slow split, all passing |
| pyproject.toml | Complete | Metadata, deps, entry points, optional groups |
| Backend abstraction | Complete | MLX default + CoreML option, `--backend` flag |
| JSON manifest contract | Stable | Structured output that downstream agents consume |
| Novel research | Documented | Object-aware tiling, Florence-2 tiling discovery, 14-model benchmark |
| No competition | True | No other open-source macOS SoM pipeline exists |

**The product works.** Packaging, trust signals, and community onboarding are now shipped.

---

## Gaps — Status

### Tier 1: "Can I trust this?" — ALL RESOLVED
- ~~No CI badge~~ — CI badge in README (Sprint 1)
- ~~No example output~~ — Hero image + manifest (Sprint 1)
- ~~README perf numbers outdated~~ — Updated (Sprint 1 + post-sprint fix)
- ~~No PyPI package~~ — `pip install uitag` works (Sprint 2 + pip fix)

### Tier 2: "Can I contribute?" — ALL RESOLVED
- ~~No CONTRIBUTING.md~~ — Added (Sprint 3)
- ~~No pre-commit config~~ — ruff check + format (Sprint 1)
- ~~No issue templates~~ — Bug report + feature request (Sprint 2)
- ~~No GitHub release~~ — v0.2.0 published (Sprint 2)

### Tier 3: "Can I build on this?" — ALL RESOLVED
- ~~No Python API docs~~ — `docs/api.md` with functions, types, manifest schema (v0.3.1)
- ~~No integration example~~ — `use_as_library.py` + `custom_backend.py` (Sprint 3)
- ~~Manifest schema not formal~~ — `uitag/schema.json` with tests (Sprint 3)

---

## Execution Plan

### Sprint 1: Trust Signals — COMPLETE (2026-02-27)

_Goal: A visitor landing on GitHub instantly sees this is real, tested, and fast._

- [x] **S1.1:** GitHub Actions CI — `.github/workflows/test.yml`, Python 3.10/3.11/3.12, badge in README
- [x] **S1.2:** Example output — hero image + redacted manifest in `docs/examples/`
- [x] **S1.3:** README refresh — perf numbers, CI badge, hero image, backend docs, 53 tests
- [x] **S1.4:** Pre-commit config — ruff check + format, codebase normalized

### Sprint 2: Distribution — COMPLETE (2026-02-27)

_Goal: `pip install uitag` works._

- [x] **S2.1:** PyPI publish — v0.2.2 live on PyPI + TestPyPI, trusted publishers (OIDC), tag-triggered workflow
- [x] **S2.2:** GitHub Release — [v0.2.0](https://github.com/swaylenhayes/uitag/releases) published
- [x] **S2.3:** Issue templates — bug report + feature request (YAML forms)

### Sprint 3: Community — COMPLETE (2026-02-27)

_Goal: Someone can contribute without asking you how._

- [x] **S3.1:** CONTRIBUTING.md — setup, architecture overview, PR workflow, welcome contributions
- [x] **S3.2:** Integration examples — `examples/use_as_library.py` + `examples/custom_backend.py`
- [x] **S3.3:** Manifest JSON Schema — `uitag/schema.json` (Draft 2020-12) + 5 validation tests

### Post-Sprint Fixes (2026-02-27)

- [x] Pip install fix — Swift source bundled in package (`uitag/tools/`)
- [x] `run_pipeline` exported from `__init__.py`
- [x] README Quick Start leads with `pip install uitag`
- [x] Apache-2.0 SPDX headers → MIT (5 files)
- [x] Stale perf numbers updated in `docs/research.md`

---

## Post-Launch (v0.3.x)

Shipped after the initial flywheel sprints:

- [x] **Batch CLI** (`uitag batch <dir>`) — Process folders of screenshots in one command (v0.3.1)
- [x] **Benchmark CLI** (`uitag benchmark <image>`) — Per-stage timing with stats across N runs (v0.3.1)
- [x] **Per-stage timing instrumentation** — All 6 pipeline stages timed in manifest output (v0.3.1)
- [x] **API reference docs** — `docs/api.md` with functions, types, manifest schema
- [x] **Performance docs** — `docs/performance.md` with stage breakdown, backend comparison
- [ ] **Pipeline architecture visual** — Replace ASCII diagram in README with a proper visual (SVG or image). The ASCII version works but doesn't convey the pipeline flow as clearly as a diagram would for first-time visitors.

---

## What We're NOT Doing (Parked)

| Item | Why parked | Resume trigger |
|------|-----------|----------------|
| CoreML as AUTO default | MLX is faster on idle GPU | Orchestration layer decides context |
| GPU load detection in selector | Orchestration-layer concern, not pipeline | Inter-agent architecture review |
| Florence-2 task token exploration (F1.4) | Shipping > features | After flywheel is turning |
| Parallel quadrant inference (F1.5) | Sequential is already 650ms | User demand |
| Additional model support (F2.0) | Scope creep risk | Community requests |

---

## Success Metrics

How we know the flywheel is turning:

| Signal | Target | Timeframe |
|--------|--------|-----------|
| GitHub stars | 25+ | 4 weeks post-launch |
| PyPI installs | 100+ | 4 weeks post-publish |
| First external issue | 1+ | 2 weeks |
| First external PR | 1+ | 4 weeks |
| README "Used by" examples | 1+ project | 8 weeks |

These are directional, not commitments. The point is to measure whether the flywheel is turning, not to hit specific numbers.

---

_This is a living document. Update as the project evolves._
