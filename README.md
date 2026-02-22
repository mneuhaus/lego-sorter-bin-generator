# lasercut-generator

Generate laser-cut DXF/SVG files (including overlap verification views) from STEP models.

## Joint Engine

Joint booleans can be executed with either backend:

1. `--joint-engine cadquery` (default): OCC/CadQuery face booleans
2. `--joint-engine shapely`: legacy Shapely boolean path

## Overlap Baseline + Heatmap Diff

Use this to catch visual regressions between runs.
By default overlap panels use `--svg-verify-overlap-mesh-offset 0.0` (edge-aligned).
Use `--svg-verify-overlap-mesh-offset <mm>` only when you explicitly want a
fold-depth shifted preview.

1. Create/update baseline from current overlap verifier output:

```bash
uv run python -m lasercut.cli step_files/bin_third_center.step \
  --format svg \
  --output output \
  --svg-overlay-original \
  --svg-verify-overlap \
  --svg-verify-overlap-baseline output/baselines/bin_third_center-overlap.svg \
  --svg-verify-overlap-update-baseline
```

2. Compare current run against baseline and generate heatmap diff artifacts:

```bash
uv run python -m lasercut.cli step_files/bin_third_center.step \
  --format svg \
  --output output \
  --svg-overlay-original \
  --svg-verify-overlap \
  --svg-verify-overlap-baseline output/baselines/bin_third_center-overlap.svg \
  --svg-verify-overlap-diff
```

Optional shifted preview example:

```bash
uv run python -m lasercut.cli step_files/bin_third_center.step \
  --format svg \
  --output output \
  --svg-verify-overlap \
  --svg-verify-overlap-mesh-offset 3.2
```

Generated files:

1. `output/lasercut-verify-overlap-diff-heatmap.png` - raw heatmap of differences
2. `output/lasercut-verify-overlap-diff-overlay.png` - current overlap view with diff overlay
3. `output/lasercut-verify-overlap-diff-summary.json` - changed-pixel stats
