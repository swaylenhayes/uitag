# uitag — Roadmap

_Last updated: 2026-03-07_
_Status: v0.4.1 current._

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Detection pipeline | Stable | 1920x1080 → 151 detections in ~2.5s accurate, ~1.7s fast |
| CLI | v0.4.1 | `uitag`, `uitag batch`, `uitag benchmark`, `uitag patch`, `uitag render` |
| OCR rescan | Shipped | Multi-crop ensemble, interactive prompt, `-rescan` suffix, special-char guard |
| Backend abstraction | Complete | MLX default + CoreML option, `--backend` flag |
| JSON manifest | Stable | Schema frozen through v0.4.x |
| Python library API | Exported | `from uitag import run_pipeline` — supports rescan, backend selection |
| Annotation rendering | v0.4.1 | Markers outside bboxes, contrast-aware text, dark gold replaces yellow |
| Test suite | 94 passing | All passing (11 skipped — require model/macOS) |
| MIT license | Done | No AGPL contamination |

---

## Up Next

- [ ] **Broader screenshot testing** — Run pipeline on IDE, settings UIs, web apps in both light/dark mode. Establish a detection quality baseline with documented expected vs. actual element counts.
- [ ] **OCR correction baseline** — Minimal deterministic heuristics for common OCR confusions in UI text (`l`→`I`, Cyrillic `Т`→`T`, `w`→`W`).

---

## Future

- [ ] **Temp file I/O optimization** — Eliminate ~600ms overhead from mlx_vlm's file-based API per quadrant call.
- [ ] **`crop_region` parameter** — Optional sub-image detection with automatic coordinate offset.
- [ ] **Manifest stability doc** — Document the schema freeze for v0.4.x in `docs/api.md`.

---

## Parked

| Item | Why parked | Resume trigger |
|------|-----------|----------------|
| CoreML as AUTO default | MLX is faster on idle GPU (benchmarked) | Profiling on M3/M4 shows otherwise |
| GPU load detection in selector | Not needed for single-user CLI | Multi-process use cases |
| Florence-2 task token exploration | Current task tokens work well | Quality issues surface |
| Parallel quadrant inference | Sequential is already ~650ms | User demand |
| Additional model support | Scope creep risk | Community requests |

---

## Completed

### v0.4.1 (2026-03-07)

- Rescan special-character guard
- CLI UX overhaul — interactive rescan prompt, bold low-confidence callout
- `-rescan` output filename suffix
- Dark gold bbox color for light-mode visibility
- Test suite fixes and lint cleanup

### v0.4.0 (2026-03-06)

- Multi-crop ensemble OCR rescan (`--rescan`)
- Patch JSON input (`uitag patch`)
- Render from manifest (`uitag render`)

### v0.3.x

- Batch CLI (`uitag batch`)
- Benchmark CLI (`uitag benchmark`)
- Per-stage timing, annotation improvements

### v0.2.x

- CI + documentation, PyPI distribution, contributor experience

---

_This is a living document. Update as the project evolves._
