# lasercut-generator

Generate laser-cuttable SVG panel layouts from STEP models for LEGO sorter bins.

The project is intended to:
- load a bin body from a STEP file
- extract individual panels such as `bottom`, `back_wall`, `right_wall`, `front_wall`, and gussets
- detect shared seams between those panels
- apply finger joints, through-slots, and optionally living hinges where appropriate
- export unfolded or packed SVG layouts for cutting
- provide a small web UI for previewing and generating those outputs

## Status

This project is actively being debugged and refined.

What is already useful:
- CLI generation from STEP to SVG
- web UI for generating outputs from files in `step_files/`
- packed and unfolded export layouts
- kerf compensation, material thickness, sheet size, and packing controls
- a strict debug pipeline for visually inspecting extraction, seams, and joint application stage by stage

What is still under active work:
- seam classification for some single-solid STEP inputs
- the transition from cleaned panel geometry to truly join-ready geometry
- some special cases around notch lips / inset back walls / through-slot seams

If you are trying to debug geometry, use the debug pipeline first before trusting final SVG output.

## Repository Layout

```text
src/lasercut/
  cli.py                Command-line entry point
  web.py                FastAPI web application
  panels.py             STEP loading, panel extraction, shared edge detection
  joints.py             Finger joints, through-slots, living hinges
  exporter.py           SVG projection, unfolding, packing
  debug_pipeline.py     Stage-by-stage visual debugging output
step_files/             Input STEP files used by CLI and web UI
output/                 Generated SVG/debug output
Dockerfile              Container image for the web app
docker-compose.traefik.yml
Handoff.md              Detailed current project state / debugging history
```

## Requirements

- Python `3.12+`
- `uv` for dependency management
- A working OpenGL-compatible environment for CadQuery in local runs
- Docker/OrbStack if you want to run the web app in a container
- Optional: a Traefik network named `traefik` for the provided compose setup

## Installation

### Local development

```bash
uv sync
```

This installs:
- `cadquery`
- `cq-warehouse`
- `fastapi`
- `uvicorn`
- `shapely`
- `svgwrite`

### Docker

Build the web app image:

```bash
docker compose -f docker-compose.traefik.yml build
```

Run it:

```bash
docker compose -f docker-compose.traefik.yml up -d
```

The provided compose file expects an external Docker network named `traefik`.

## How It Is Supposed To Work

At a high level, the intended processing pipeline is:

1. Load a STEP model.
2. Extract panel faces for each wall/bottom/gusset.
3. Rebuild those panels as separate solids with known thickness.
4. Detect shared seams between panels.
5. Classify each seam as one of:
   - normal finger joint
   - through-slot seam
   - living hinge seam
6. Apply joint geometry to the panel solids.
7. Project the result to 2D and export:
   - `unfolded` layout
   - `packed` layout

### Important current nuance

For difficult single-solid STEP inputs, the project now distinguishes between:
- **raw extracted geometry**
- **clean/reference geometry**
- **seam classification**
- **joint application**

This matters because a shape can look good in raw/clean extraction and still fail when joints are applied. The debug pipeline exists specifically to make those transitions visible and testable.

## CLI Usage

The main CLI entry point is:

```bash
uv run python -m lasercut <step-file>
```

Example:

```bash
uv run python -m lasercut step_files/bin_third_left.step \
  --thickness 1.8 \
  --kerf 0.02 \
  --layout unfolded \
  --output output
```

### Common options

- `--thickness` material thickness in mm
- `--finger-width` target finger/tab width in mm
- `--kerf` kerf compensation in mm
  - positive = tighter fit
  - negative = looser fit
- `--living-hinge-angle` use living hinges on shallow non-bottom seams
- `--layout unfolded|packed`
- `--sheet-width` / `--sheet-height` required for packed layout
- `--part-gap` spacing between parts in packed layout
- `--sheet-gap` spacing between sheets in exported SVG
- `--pack-rotations` number of tested rotation steps for packing
- `--output` output root directory

### Packed layout example

```bash
uv run python -m lasercut step_files/bin_third_left.step \
  --thickness 1.8 \
  --kerf 0.02 \
  --layout packed \
  --sheet-width 710 \
  --sheet-height 180 \
  --pack-rotations 8 \
  --output output
```

Generated files are written into a folder name that includes:
- thickness
- kerf
- sheet size for packed layouts

## Web UI

Run locally without Docker:

```bash
uv run uvicorn lasercut.web:app --host 0.0.0.0 --port 8000
```

The web app:
- lists STEP files from `step_files/`
- generates multiple files in parallel
- supports unfolded and packed layouts
- returns downloadable SVGs and ZIP archives
- renders 3D previews of the original STEP input

### Environment variables

- `LASERCUT_STEP_DIR`
  - override the directory used for STEP files
- `LASERCUT_WEB_JOB_TTL_SECONDS`
  - job retention time for generated results
- `LASERCUT_WEB_MAX_WORKERS`
  - max parallel generation workers

### Camera test page

There is also a helper page used for preview/camera testing:
- `camera_test.html`

It is referenced by the web app and Docker image, so it should not be deleted casually.

## Debug Pipeline

The debug pipeline is the most important tool when geometry looks wrong.

Run it like this:

```bash
uv run python -m lasercut.debug_pipeline step_files/bin_third_left.step \
  --thickness 1.8 \
  --kerf 0.02 \
  --output output/debug-pipeline
```

This generates a numbered sequence of artifacts, for example:

```text
00_raw_extract.svg
01_raw_vs_clean.svg
02_clean_geometry.svg
03_seams.svg
04_joint_application.svg
X_current_generator_comparison.svg
report.json
```

### Meaning of the stages

- `00_raw_extract`
  - first extracted panel geometry from STEP
- `01_raw_vs_clean`
  - compares raw extraction with working geometry
- `02_clean_geometry`
  - cleaned/reference geometry before joints
- `03_seams`
  - seam classification drawn on the same geometry as stage 02
- `04_joint_application`
  - joints applied directly to the numbered stage chain
- `X_current_generator_comparison`
  - separate comparison against the old/current source-based generator

### Why this exists

The project previously suffered from “invisible jumps” where a later debug stage secretly used different source geometry than the stage before it. The debug pipeline is now meant to be a strict handoff chain, so we can visually verify what each step receives and produces.

If `04_joint_application.svg` looks bad while `02_clean_geometry.svg` looks good, the failure is between seam logic and joint application, not in raw extraction.

## STEP Input Expectations

The generator works best when the STEP file is clean and consistent.

Things that help a lot:
- consistent wall thickness
- clean intersections without tiny sliver faces
- fewer ambiguous lip/notch remnants on side walls
- stable geometry around the inset/notched back wall

Things that are currently tricky:
- single-solid bodies where the outer wall face includes extra lip/notch material
- seams that visually look simple but are offset by stock thickness in the extracted geometry
- cases where the back notch/lip extends beyond the semantic wall boundary

## Naming / Semantics

Internally, the project tries to reason in terms of semantic panels:
- `bottom`
- `back_wall`
- `right_wall`
- `front_wall`
- `front_gusset`
- `back_gusset`

One especially important rule for this project domain:
- the **back wall** is the inset/notched wall on the notch-lip side

This semantic distinction has been a recurring source of bugs, so be careful when changing panel naming heuristics.

## Current Reality / Limitations

Please treat the current implementation as:
- useful for exploration and debugging
- partially useful for real SVG generation
- not yet fully reliable for all single-solid STEP geometries

The current main open engineering problem is:
- going from clean/reference panel geometry to truly join-ready geometry before joints are applied

That work is documented in detail in:
- `Handoff.md`

## Development Notes

A few important lessons from recent work:
- do not mix extraction fixes, seam fixes, exporter fixes, and web fixes in the same debugging step
- do not trust final SVG output if intermediate stages were not checked
- do not assume that a visually clean panel extract is already correct for seam classification
- prefer small, isolated changes and verify each stage visually

## Useful Commands

### Run CLI on one file

```bash
uv run python -m lasercut step_files/bin_third_left.step --thickness 1.8 --kerf 0.02
```

### Run debug pipeline on the three current third-bin files

```bash
uv run python -m lasercut.debug_pipeline step_files/bin_third_left.step --thickness 1.8 --kerf 0.02 --output output/debug-pipeline-1p8
uv run python -m lasercut.debug_pipeline step_files/bin_third_center.step --thickness 1.8 --kerf 0.02 --output output/debug-pipeline-1p8
uv run python -m lasercut.debug_pipeline step_files/bin_third_right.step --thickness 1.8 --kerf 0.02 --output output/debug-pipeline-1p8
```

### Run web app locally

```bash
uv run uvicorn lasercut.web:app --host 0.0.0.0 --port 8000
```

### Run Docker/Traefik setup

```bash
docker compose -f docker-compose.traefik.yml up -d --build
```

## Related Files

- `Handoff.md` - detailed current debugging history and next steps
- `src/lasercut/panels.py` - extraction and panel semantics
- `src/lasercut/joints.py` - seam classification and joint logic
- `src/lasercut/debug_pipeline.py` - current most important debugging tool
- `src/lasercut/exporter.py` - SVG layout/export
- `src/lasercut/web.py` - web UI

## Suggested Next Step for Future Work

If you are resuming development, the most productive next step is:

1. trust `00`–`02`
2. inspect `03`
3. add a real `join-ready geometry` stage before joint application
4. only then try to improve `04_joint_application`

That is the cleanest path out of the current geometry/seam mismatch.
