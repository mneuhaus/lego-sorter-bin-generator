# Lasercut Generator: Lessons, Fixes, and Risk Areas (2026-02-24)

## Scope
This document summarizes what we learned while stabilizing the STEP -> panel -> joinery -> SVG pipeline, including regressions, difficult areas, and recommended guardrails.

## What We Fixed
- Switched SVG presentation from red/black to black/white for readability.
- Integrated `cq_warehouse`-based finger-joint generation with custom fallback logic.
- Fixed inset seam behavior for back-wall/side-wall and bottom/back-wall intersections.
- Added original-outline overlay (green dashed) for fast visual dimension comparison.
- Added packed layout mode with sheet-size constraints.
- Added output metadata in folder and filename:
  - thickness (`3.2mm`),
  - kerf (`k0.02mm`),
  - sheet size (`710x180`).
- Added kerf as explicit option (positive kerf -> tighter fit intent).
- Improved packing and set practical default rotation sampling in web flow.
- Added Docker + Traefik web deployment and kept it rebuildable from repo state.
- Upgraded web UI with:
  - multi-select generation,
  - parallel jobs,
  - per-file previews,
  - ZIP download,
  - fixed-angle 3D STEP preview,
  - localStorage-persisted settings,
  - full-width SVG preview,
  - fullscreen SVG modal zoom.
- Added minimum required sheet-size feedback for non-fitting parts.
- Added living hinge for shallow-angle seams and evolved to merged-piece hinge output with line-slit lattice and double-cut support.
- Added 1 mm hinge seam overlap compensation in merged hinge placement.

## Hard / Risky Areas
- Inset/lip seams are fragile and non-generic; they require lip-aware slot interval extraction.
- Boolean operations that look valid in 3D can create tiny 2D contour artifacts in exported SVG.
- Joinery seam phase changes can silently flip tab/slot parity at corners.
- Through-slot and living-hinge logic can interfere through shared cleanup and projection paths.

## Regressions We Hit (and Why)

### 1) Stray corner fingers at back-wall seam
- Symptom: tiny protrusions beyond back-wall extent.
- Root cause: seam phase and cleanup interactions at back-wall/side-wall seam endpoints.
- Stabilization:
  - deterministic seam phase handling for back-wall/side-wall seams,
  - side-wall-focused seam-end trimming.

### 2) Through-slot extension reintroduced protrusions
- Symptom: old stray protrusions came back after a "tab protrusion" change.
- Root cause: fused extension boxes in through-slot wall-tab regions created contour side effects.
- Resolution: removed extension-fuse block (kept stable recess/slot path).

### 3) Living hinge geometry initially not manufacturable
- Symptom: hinge cuts not spanning full region / wrong cut style.
- Root cause: edge-local slit logic without proper merged-piece treatment.
- Resolution: merge hinge-connected panels first, then generate edge-to-edge slit lattice.

## Decisions That Worked Well
- Prefer single-solid equal-thickness STEP where possible; seam detection is cleaner.
- Keep manufacturing metadata in output names (thickness/kerf/sheet).
- Treat web deployment as part of dev loop:
  - commit,
  - push,
  - `docker compose -f docker-compose.traefik.yml up -d --build`.
- Validate geometry changes with regenerated known samples (not only code review).

## Current Known Open Questions
- Do we need explicit through-slot tab protrusion geometry, or is current flush behavior sufficient with kerf/material tuning?
- Should hinge overlap (currently 1.0 mm) be user-configurable in CLI/UI?

## Suggested Guardrails
- Keep a golden sample set (`bin_half_left`, `bin_half_right`, `bin_third_left`, `bin_full`) and regenerate on every joinery change.
- Add automated checks for:
  - micro protrusions at seam ends,
  - orphan inner loops touching boundaries,
  - unintended contour self-intersections.
- Test both generation paths after each change:
  - `cq_warehouse` path,
  - custom fallback path.

## Reference Commands
```bash
uv run python -m lasercut step_files/bin_half_left.step \
  --thickness 3.2 --finger-width 20 --living-hinge-angle 45 \
  --kerf 0.02 --layout packed --sheet-width 710 --sheet-height 180 \
  --pack-rotations 8 --output output

uv run python -m lasercut step_files/bin_third_left.step \
  --thickness 3.2 --finger-width 20 --living-hinge-angle 45 \
  --kerf 0.02 --layout packed --sheet-width 710 --sheet-height 180 \
  --pack-rotations 8 --output output
```

## Key Files
- `src/lasercut/panels.py`
- `src/lasercut/joints.py`
- `src/lasercut/exporter.py`
- `src/lasercut/cli.py`
- `src/lasercut/web.py`
- `docker-compose.traefik.yml`
- `Dockerfile`
