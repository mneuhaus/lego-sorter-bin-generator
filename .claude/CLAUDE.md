# Lasercut Generator

STEP-to-lasercut (SVG/DXF) box generator with finger joints.

## Running

Always use `uv` to run Python. Never use `python` or `python3` directly.

```bash
# Generate output
uv run python -m lasercut.cli step_files/bin_third_center.step --thickness 3 --layout folded --wall-offset 10

# All STEP models are in step_files/
```

## Previewing SVG Output

Convert SVG to PNG with `rsvg-convert`, then read the PNG to visually verify:

```bash
rsvg-convert -d 300 -p 300 output/lasercut.svg -o output/lasercut.png
```

Then use the Read tool on `output/lasercut.png` to view it inline.

Do NOT use the browser for SVG preview (extension connectivity is unreliable).

## Architecture

- `src/lasercut/cli.py` - CLI entry point
- `src/lasercut/step_loader.py` - STEP file loading via cadquery/OCC
- `src/lasercut/face_classifier.py` - Bottom/wall classification, shared edges
- `src/lasercut/projector.py` - 3D face to 2D projection
- `src/lasercut/finger_joints.py` - Fusion-style finger joint generation (the only algorithm)
- `src/lasercut/exporter.py` - SVG/DXF export with folded/packed layout

## Key Design Decisions

- **Fusion-style intervals**: Finger joints use absolute (start_mm, width_mm) intervals, not parametric fractions
- **Morphological cleaning**: Buffer at 1x thickness (not 1.5x) to preserve arcs while removing ledge artifacts
- **Fixed outer dimensions**: Walls grow inward only; outer box envelope stays constant regardless of material thickness
- **Corner keepouts**: `depth * cot(interior_angle/2)` clearance at convex corners
