# uitag — Roadmap

_Last updated: 2026-03-08_
_Status: Sprints 1-3 complete. v0.3.1 shipped (batch, benchmark). v0.4.1 current (rescan, patch, render, CLI UX polish)._

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
| Annotation rendering | Improved | Markers outside bboxes, contrast-aware text colors (2026-03-06) |

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

### Annotation & CLI Improvements (2026-03-06)

- [x] **Marker repositioning** — SoM numbered circles positioned outside bbox (above-left) to avoid occluding UI text
- [x] **Contrast-aware text** — Marker text auto-switches black/white based on background luminance (yellow, orange, cyan get black text)
- [x] **Resolved output paths** — CLI prints absolute paths instead of `./` for output location
- [x] **Diff render prototype** — 9 iterations of instruction overlay rendering, documented in `docs/diff-render-spec.md`
- [x] **PDF companion prototype** — Selectable text PDF alongside instruction images

---

## Completed — v0.4.0

### Feature A: Multi-Crop Ensemble OCR Rescan

**Spec:** `docs/specs/multi-scale-ocr-rescan.md` | **Research:** `docs/research/ocr-rescan-experiments.md` | **Release:** `docs/releases/v0.4.0.md`

- [x] Implement multi-crop ensemble re-OCR pipeline stage (`uitag/rescan.py`)
- [x] Add `--no-lang-correction` flag to Swift binary
- [x] Add `--rescan` and `--rescan <ids>` CLI flags
- [x] Add low-confidence callout to default CLI output
- [x] Confidence threshold set to 0.8
- [x] 8-phase experiment validating approach (crop sensitivity, context, light/dark mode)
- [x] Light mode OCR advantage documented in README and research
- [ ] Determine verbose vs. non-verbose output behavior (TBD)
- [ ] Determine batch output format for low-confidence elements (TBD)

### Feature B: Patch JSON Input (Re-Annotation)

**Spec:** `docs/specs/patch-json-input.md` | **Release:** `docs/releases/v0.4.0.md`

- [x] Define and validate patch JSON schema (`uitag/patch.py`)
- [x] Implement `uitag patch` subcommand (`uitag/patch_cli.py`)
- [x] Implement `uitag render` subcommand (manifest-to-image, no detection)
- [x] Output naming: `{stem}-uitag.png` + `{stem}-uitag-manifest.json`
- [x] Partial patches: unpatched elements pass through unchanged

---

## Up Next — P1

- [ ] **Pipeline architecture visual** — Replace ASCII diagram in README with SVG/image
- [ ] **OCR correction baseline** — Minimal deterministic heuristics for common OCR confusions in UI text (see `docs/ocr-correction-strategy.md`)

---

## Up Next — Validation

- [ ] **Dense UI testing** — Test diff render with many changes in a small area
- [x] **Light mode validation** — Light mode produces measurably better OCR for special characters (v0.4.0 research)
- [x] **Dark mode validation** — Dark mode baseline established; thin characters (`\`) unrecoverable in some positions (v0.4.0 research)
- [ ] **Broader screenshot testing** — IDE, settings UIs, web apps in both light/dark mode
- [ ] **Large callout count** — Test stacking/overlap with 5+ callout boxes
- [ ] **Long values** — Test truncation or wrapping for very long field values
- [x] **Dark mode detection** — CLI hint when dark mode screenshots may benefit from light mode recapture (v0.4.1)

---

## Handed Off to Agent Layer

| Item | Why | Handoff doc |
|------|-----|-------------|
| PDF user-guidance generation | Requires contextual judgment, narrative, layout decisions | `docs/specs/pdf-generation-handoff.md` |
| LLM post-correction calls | Requires inference, model selection, cost management | `docs/ocr-correction-strategy.md` |
| Multi-step correction orchestration | Requires decision-making about when to rescan, escalate | `docs/ocr-correction-strategy.md` |

---

## Under Debate

| Item | Question | See |
|------|----------|-----|
| Curated domain dictionaries | UITag core, premium package, or agent layer? | `docs/ocr-correction-strategy.md` |
| Comprehensive regex patterns | Same boundary question | `docs/ocr-correction-strategy.md` |
| Who maintains curated data? | Open-source community, paid service, or agent? | `docs/ocr-correction-strategy.md` |

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
