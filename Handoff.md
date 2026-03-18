# Handoff: Lasercut Generator Debugging / Third-Bin STEP Pipeline

## Session Metadata
- Created: 2026-03-18 18:45:40 CET
- Project: `/Users/mneuhaus/Workspace/LegoSorter/lasercut-generator`
- Branch: `main`
- Current committed HEAD: `fc2e368` (`wip: split back lip from single-solid join geometry`)
- Last stable remote main reference repeatedly used during rollback/tests: `4bddb05`

## Current State Summary
We spent a long time fighting geometry extraction, naming, seam classification, and joint generation for laser-cut bin panels derived from STEP files. The biggest recurring pain point has been single-solid STEP input where the bottom notch/lip bleeds into wall extraction, especially around `back_wall` and `right_wall`, causing stray tabs/fingers, incorrect seam classification, and misleading final SVGs. The biggest recent improvement is that there is now a real debug pipeline that produces visible stage-by-stage artifacts and an explicit stage manifest. Numbered debug stages are now chained honestly from one to the next, and the old source-based generator output has been moved out of the numbered sequence into a comparison artifact. The pipeline still breaks at joint application, but the break is now localized and visible instead of hidden.

## What Was Tried So Far

### Earlier broad work before the current debugging push
- Switched the display theme from red/black to black/white because the original scheme was hard to inspect visually.
- Fixed a number of earlier missing finger/cutout issues for `right_wall`, `back_wall`, `front_wall`, and `back_gusset` in older STEP variants.
- Explored using `cq_warehouse` / finger-jointed box style logic and a full-thickness single-body workflow instead of separated STEP bodies.
- Added packed layout support with sheet size, kerf, thickness, filename/folder naming, and web UI support.
- Added web UI features like multiple generation, previews, ZIP download, localStorage settings, etc.
- Added living hinge support for shallow-angle seams and iterated on slit patterns and widths.
- Built and deployed a local Docker + Traefik setup for the web UI at `http://lego-sorter-bin-generator.traefik.me/`.

### Why that earlier progress kept regressing
We were making changes on several layers at once:
- extraction / panel semantics
- seam classification
- special-case joint logic
- exporter / SVG projection
- web rendering

That made it too easy to “fix” a display symptom in the wrong layer and re-break something else.

### Reset / rollback history that mattered
- Repeatedly rolled back to `bdbff82` and `4bddb05` to recover from increasingly bad geometry experiments.
- Determined that older commits did not magically fix the core issue; cleaner STEP input helped more than older code.
- The current committed codebase is on `fc2e368`, but there are important uncommitted debugging changes in `src/lasercut/debug_pipeline.py` and some ongoing `src/lasercut/panels.py` changes.

## Important Technical Reasoning and Lessons Learned

### 1. The STEP input itself matters a lot
We repeatedly thought a code bug was the cause when the STEP geometry itself was still not ideal. Several newer STEP files improved generation substantially.

Current relevant STEP files in worktree:
- `step_files/bin_third_left.step`
- `step_files/bin_third_center.step`
- `step_files/bin_third_right.step`

These are now treated as the 1.8mm-base set.

### 2. Single-solid outer-face extraction is the root of many artifacts
For single-solid STEP files, extracting a wall from the **outer** face of the solid pulls in lip/notch geometry that semantically belongs to the bottom-lip region, not the wall itself. This is why `right_wall` and sometimes `back_wall` showed stray “fingers” in raw geometry.

The strong current conclusion is:
- For walls/gussets, the **inner/opposite face** is the cleaner reference geometry.
- For `bottom`, the **outer/lower face** is still the better raw extract because it preserves the true footprint.

This hybrid approach is now reflected in debug stage `00_raw_extract`.

### 3. Clean geometry is not automatically join-ready geometry
This is one of the most important discoveries.

`00`–`02` now look good because they represent a cleaned/reference geometry. However, seam/joint logic still expects geometry closer to actual join boundaries. When seams were compared against the clean geometry, the wall-side seam positions were often offset by roughly the original stock thickness.

Measured pattern (example from the 1.8mm set):
- bottom-side seam alignment often approximately `0.0 mm`
- wall/gusset seam alignment often approximately `1.4–1.9 mm`
- this strongly suggests the clean geometry is a **reference/interior geometry**, not yet the final join-ready geometry

This means the real missing stage is likely:
- `03b_join_ready_geometry` or similar
- where relevant edges are compensated/expanded from clean reference geometry to actual join lines before joint application

### 4. Debug stages were misleading until recently
A major source of confusion was that the earlier debug pipeline was not truly linear:
- `00–02` were showing cleaned geometry
- `04` jumped back to the old source-based generator path
- so the numbered stages did **not** actually represent a real build pipeline

This has now been corrected.

## Current Debug Pipeline Design

The new numbered debug stages now represent a real handoff chain:

1. `00_raw_extract`
- Initial extracted shapes from STEP
- Walls/gussets use inner-face extraction
- Bottom uses outer/lower face extraction
- Green dashed overlay shows old outer-face raw extraction for comparison

2. `01_raw_vs_clean`
- Builds from stage `00`
- Recomputes shared edges from stage `00` panel solids
- Black = working geometry, green dashed = raw basis

3. `02_clean_geometry`
- Canonical cleaned/reference geometry before seams and joints
- This stage currently looks good and is the best visual foundation so far

4. `03_seams`
- Seam classification built from stage `02` geometry only
- Drawn on the same stage `02` outlines
- No fallback to source-model seams in the numbered stage chain
- Each seam pair gets a unique color, pair labels, and seam type indication

5. `04_joint_application`
- True sequential continuation of stage `03`
- Joints are applied directly to the stage `02/03` geometry with `source_solid=None`
- This stage is currently still bad, but it is at least honest: if it fails, the failure is between stage `03` and `04`, not because of a hidden source-model jump

Separate comparison artifact:
- `X_current_generator_comparison.svg`
- This is the old/current source-based generator output and is intentionally **not** part of the numbered debug pipeline

## Current Debug Artifacts

For the current 1.8mm set, debug artifacts exist here:
- `output/debug-pipeline-1p8/bin_third_left-t1.8-k0.02/`
- `output/debug-pipeline-1p8/bin_third_center-t1.8-k0.02/`
- `output/debug-pipeline-1p8/bin_third_right-t1.8-k0.02/`

Each folder now contains:
- `00_raw_extract.svg`
- `01_raw_vs_clean.svg`
- `02_clean_geometry.svg`
- `03_seams.svg`
- `04_joint_application.svg`
- `X_current_generator_comparison.svg`
- `report.json`

Important:
- Older runs may still have stale `04_current_generator_joints.svg` / `05_pipeline_attempt_joints.svg` files in old folders.
- The current pipeline should be treated as the numbered stages above plus the single `X_...` comparison artifact.

## Current State of the Three 1.8mm Files

### `bin_third_left.step`
Panels:
- `bottom`
- `right_wall`
- `back_wall`
- `front_gusset`
- `front_wall`

Current seam classification in `03_seams`:
- `S1 bottom<->right_wall : through_slot`
- `S2 bottom<->back_wall : through_slot`
- `S3 bottom<->front_gusset : through_slot`
- `S4 bottom<->front_wall : through_slot`
- `S5 right_wall<->back_wall : finger`
- `S6 back_wall<->front_wall : finger`
- `S7 front_gusset<->front_wall : living_hinge`

Observation:
- `00–02` look decent.
- `03` is now at least honest and visually useful.
- `04` still fails because too many `bottom-*` seams are classified as `through_slot`.

### `bin_third_center.step`
Panels:
- `right_wall`
- `front_wall`
- `bottom`
- `back_wall`

Current seam classification in `03_seams`:
- `S1 right_wall<->bottom : through_slot`
- `S2 right_wall<->back_wall : finger`
- `S3 front_wall<->bottom : through_slot`
- `S4 front_wall<->back_wall : finger`
- `S5 bottom<->back_wall : through_slot`

Observation:
- Same pattern: wall-to-bottom seams are still being misclassified as through-slots.

### `bin_third_right.step`
Panels:
- `bottom`
- `front_wall`
- `back_wall`
- `back_gusset`
- `right_wall`

Current seam classification in `03_seams`:
- `S1 bottom<->front_wall : through_slot`
- `S2 bottom<->back_wall : through_slot`
- `S3 bottom<->back_gusset : through_slot`
- `S4 bottom<->right_wall : through_slot`
- `S5 front_wall<->back_wall : finger`
- `S6 back_wall<->right_wall : finger`
- `S7 back_gusset<->right_wall : living_hinge`

Observation:
- Same pattern as the others.

## Key Current Conclusion
The pipeline is no longer “lying”, but the seam model is still wrong for clean/reference geometry.

Specifically:
- `00` / `01` / `02` are acceptable
- `03` is now a real seam classification on the same geometry used by the numbered pipeline
- `04` is bad because the seam classification / joint classification is not yet compatible with the clean/reference geometry

In other words:
- the next real work is **not** to keep patching final SVGs
- the next work is to define a proper **join-ready geometry** between `02` and `03/04`

## Files Modified in Current Uncommitted Work

### `src/lasercut/debug_pipeline.py`
Main ongoing debug work. Important recent changes:
- introduced a real numbered stage chain
- moved old generator output to `X_current_generator_comparison.svg`
- stage manifest / dependency metadata added to `report.json`
- `00_raw_extract` uses inner-face extraction for walls and outer-face extraction for bottom
- `03_seams` now classifies from the clean stage model rather than the source model
- `04_joint_application` now truly consumes the stage-02/03 geometry instead of jumping back to source geometry

### `src/lasercut/panels.py`
Still modified in working tree. Important context:
- `load_step_panels(...)` currently has `enable_back_lip_split=False` by default
- several experiments were done here earlier around virtual back-lip splitting and cleanup
- for the current debugging direction, the numbered pipeline is not relying on that lip split

### STEP files changed in worktree
- `step_files/bin_third_left.step`
- `step_files/bin_third_center.step`
- `step_files/bin_third_right.step`

These should be treated as intentional user-provided geometry updates.

## Decisions Made and Why

### Use inner-face extraction for walls, outer-face extraction for bottom in raw debug stage
Reason:
- This removed lip-induced wall artifacts while preserving bottom footprint.

### Stop pretending the numbered debug stages are one chain if they are not
Reason:
- This had become one of the biggest sources of confusion and frustration.
- Honest but ugly output is better than a misleading “nice” pipeline.

### Move current source-based generator output to a comparison artifact
Reason:
- It is still useful to compare against current production behavior.
- It must not appear as step `04` in a supposedly sequential pipeline.

### Do not keep broad geometry/export/joint changes entangled
Reason:
- Previous regression cycles repeatedly came from touching the wrong abstraction layer to fix a symptom.

## Tests and Commands Run

### Main debug pipeline command
```bash
uv run python -m lasercut.debug_pipeline step_files/bin_third_left.step --thickness 1.8 --kerf 0.02 --output output/debug-pipeline-1p8
uv run python -m lasercut.debug_pipeline step_files/bin_third_center.step --thickness 1.8 --kerf 0.02 --output output/debug-pipeline-1p8
uv run python -m lasercut.debug_pipeline step_files/bin_third_right.step --thickness 1.8 --kerf 0.02 --output output/debug-pipeline-1p8
```

### Useful inspection commands
```bash
git status --short
git log --oneline --decorate -n 12
rg -n "debug_pipeline|load_step_panels|apply_finger_joints" src/lasercut -S
```

## Current Git State
At the time of writing:
```text
 M src/lasercut/panels.py
 M step_files/bin_third_center.step
 M step_files/bin_third_left.step
 M step_files/bin_third_right.step
?? Archiv.zip
?? src/lasercut/debug_pipeline.py
```

Important:
- `src/lasercut/debug_pipeline.py` is **uncommitted** and contains the new real stage-chain work.
- `Archiv.zip` is unrelated and should not be accidentally committed.
- The STEP files are user changes and should be treated carefully.

## Blockers / Open Problems

### 1. Missing `join-ready geometry` stage
This is the main blocker.

Right now:
- `02` is a clean/reference geometry
- but it is not yet the geometry that seam logic expects
- so `03` / `04` still push too many `bottom-*` seams into `through_slot`

Likely next required stage:
- `03b_join_ready_geometry`
- explicitly transform/offset the clean/reference geometry into true join-boundary geometry before seam classification or jointing

### 2. Through-slot classification is still too aggressive on clean/reference geometry
Affected seams across the 1.8mm set:
- `bottom<->right_wall`
- `bottom<->front_wall`
- `bottom<->front_gusset` / `bottom<->back_gusset`
- `bottom<->back_wall`

### 3. Living hinge logic still exists and may or may not be desired
Current living-hinge seams:
- `front_gusset<->front_wall` on left
- `back_gusset<->right_wall` on right

This is not the main blocker right now, but it is still active in seam classification / joint logic.

## Immediate Next Steps

1. Keep the numbered debug pipeline as the single source of truth.
- Do not reintroduce source-model jumps into `00`–`04`.

2. Introduce an explicit `join-ready geometry` stage between `02` and seam/joint application.
- This should take the clean/reference geometry and compensate relevant edges toward the actual join lines.
- Only after this stage should seam classification be considered authoritative.

3. Re-run the debug pipeline on all three 1.8mm files after that new stage exists.
- The first success criterion is not “pretty joints” yet.
- The first success criterion is: bottom-side seams stop incorrectly collapsing into many `through_slot` classifications.

4. Only after the numbered pipeline is trustworthy again, revisit the production generator.
- Production output should be adapted to follow the proven stage chain, not the other way around.

## Things To Avoid Repeating
- Do not “fix” a bad final SVG by immediately changing exporter projection logic.
- Do not mix geometry cleanup, seam classification, and joint application changes in the same debugging step.
- Do not present comparison artifacts as if they are part of the numbered pipeline.
- Do not assume `02_clean_geometry` is already join-ready.

## Environment / Runtime Notes
- Local dev domain: `http://lego-sorter-bin-generator.traefik.me/`
- There is a global Traefik stack at `/Users/mneuhaus/Workspace/traefik-global/docker-compose.yml`
- Previous OrbStack reset caused daemon/network issues; those were repaired and a global Traefik was brought back up

## Critical Files
- `src/lasercut/panels.py`
  - STEP loading, panel extraction, raw geometry semantics
- `src/lasercut/joints.py`
  - seam classification, finger joints, through-slot logic, living hinge logic
- `src/lasercut/debug_pipeline.py`
  - current most important debugging surface
- `src/lasercut/exporter.py`
  - SVG projection/export
- `src/lasercut/web.py`
  - web generation entrypoint
- `output/debug-pipeline-1p8/*/report.json`
  - current factual stage/seam summaries for the three 1.8mm files

## Final Honest Summary
The current best progress is not that the generator is fixed. It is that the debugging surface is finally becoming trustworthy.

What is solid now:
- raw extraction direction is better understood
- `00`–`02` are reasonably good
- numbered debug stages now represent a real sequential pipeline
- the old generator is no longer masquerading as part of that pipeline

What is not solved yet:
- the transition from clean/reference geometry to seam-ready and joint-ready geometry
- therefore `04_joint_application.svg` is still not a usable result

That is the next real engineering problem.
