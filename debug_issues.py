"""Debug the specific issues with the rewritten parametric.py."""
import sys
sys.path.insert(0, 'src')

import math
from lasercut.parametric import (
    _extract_planar_faces, _identify_structural_faces, _compute_thickness_from_faces,
    _unit, _cross3,
)


# bin_full analysis
print("=" * 70)
print("bin_full: Bottom panel analysis")
print("=" * 70)

faces_full = _extract_planar_faces('step_files/bin_full.step')
sf_full = _identify_structural_faces(faces_full)

print(f"Inner BR: ({sf_full.inner_br[0]:.2f}, {sf_full.inner_br[1]:.2f}, {sf_full.inner_br[2]:.2f})")
print(f"Inner RT: ({sf_full.inner_rt[0]:.2f}, {sf_full.inner_rt[1]:.2f}, {sf_full.inner_rt[2]:.2f})")
print(f"Inner BL: ({sf_full.inner_bl[0]:.2f}, {sf_full.inner_bl[1]:.2f}, {sf_full.inner_bl[2]:.2f})")
print(f"Y front: {sf_full.y_front:.2f}")
print(f"Y back: {sf_full.y_back:.2f}")

# The issue: inner BL is (-38.93, -383.75, 140.95) which is the simple BL corner
# But the bottom outer face extends much further left (to -151.11) due to gussets.
# For the bottom panel, we need the FULL extent.

# Let's look at what the bottom outer face vertices show
bo = sf_full.bottom_outer
print(f"\nBottom outer vertices ({len(bo.vertices)}):")
bo_xs = sorted(set(round(v[0], 1) for v in bo.vertices))
print(f"  Unique X values: {bo_xs}")
# The leftmost X on the bottom outer is around -151.11 (gusset vertex)
# The rightmost is 39.15

# The key issue: for bin_full, the bottom panel should extend to the gusset area.
# But for the parametric model, we want a SIMPLE rectangle, not the complex gusset shape.
# The gusset area is where the left wall triangles attach.

# For bin_full with no left wall: the bottom extends to x=-39.53 at front/back,
# and to x=-151.11 in the middle (gusset peaks).
# For laser cutting, the bottom panel would be the full rectangular extent from
# (39.15, 94.49) to (-39.53, 139.91) -- the simple part.

# But the STEP shows the bottom as ONE face including the gusset triangles.
# The gusset triangles are separate triangular faces with different normals though.
# So the bottom outer face includes gusset geometry embedded as vertices.

# Looking at the bottom inner face: it's also not a simple rectangle.
# It has 10 vertices including gusset notches.

print(f"\nBottom inner vertices ({len(sf_full.bottom_inner.vertices)}):")
for v in sf_full.bottom_inner.vertices:
    print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")

# Inner bottom vertices:
# (-38.93, -383.75, 140.95) -- BL at front Y
# (35.61, -383.75, 97.91)   -- BR at front Y
# (35.61, -82.73, 97.91)    -- BR at back Y
# (-38.93, -82.73, 140.95)  -- BL at back Y
# (-150.01, -146.86, 205.08) -- gusset peak (back half)
# (-134.58, -223.24, 196.17) -- gusset middle
# (-134.58, -232.04, 196.17) -- gusset middle
# (-134.58, -234.44, 196.17) -- gusset middle
# (-134.58, -243.24, 196.17) -- gusset middle
# (-150.01, -319.61, 205.08) -- gusset peak (front half)

# For the "simple rectangle" bottom:
# BR is at x=35.61 (inner), z=97.91 (inner)
# BL is at x=-38.93 (inner), z=140.95 (inner)
# These are the 4 corners of the rectangular part.
# Width = dist((35.61, 97.91), (-38.93, 140.95)) = sqrt(74.54^2 + 43.04^2) = 86.07mm

# For bin_full, the bottom OUTER face rightmost vertex is at x=39.15, z=94.49
# and the leftmost simple vertex is at x=-39.53, z=139.91 (from end wall vertices).

# So the outer bottom rectangle goes from (39.15, 94.49) to (-39.53, 139.91)
# Width = sqrt((39.15+39.53)^2 + (94.49-139.91)^2) = sqrt(78.68^2 + 45.42^2) = 90.85mm

# But the geometry_params.py says BOTTOM_LENGTH = 201.30mm!
# That's the full bottom including gusset extensions.

# Wait, let me re-read geometry_params.py more carefully:
# BOTTOM_LENGTH = 201.30 "mm, outer edge of bottom panel in cross-section"
# That's the LENGTH along the bottom surface, not just the "simple" part.

# The bottom outer face goes from (39.15, 94.49) on the right to (-151.11, 204.33) on the left.
# Distance = sqrt((39.15+151.11)^2 + (94.49-204.33)^2) = sqrt(190.26^2 + 109.84^2) = 219.65mm
# That's not 201.30 either.

# Hmm, for the parametric model, the bottom panel dimension depends on the
# cross-section at a particular Y position. The gusset changes the extent.

# At the front/back ends: bottom goes from (39.15, 94.49) to (-39.53, 139.91) = 90.85mm
# At the gusset peaks: extends to (-151.11, 204.33) = much wider

# The existing geometry_params says:
# bin_full: bottom_length_at_mid = 201.30mm
# bin_third_left: bottom_length_at_mid = 211.47mm

# These seem to be the bottom length at the Y midpoint.
# For bin_full at the midpoint, the bottom extends from BR (39.15, 94.49)
# toward the gusset. Let me check what vertex the bottom has near the Y midpoint.

mid_y = (sf_full.y_front + sf_full.y_back) / 2
print(f"\nY midpoint: {mid_y:.2f}")
# Y midpoint of bin_full: (-385.09 + -81.39) / 2 = -233.24

# The bottom outer face vertices near Y=-233.24:
# (-135.18, -243.24, 195.13)
# (-135.18, -234.44, 195.13)
# (-135.18, -232.04, 195.13)
# (-135.18, -223.24, 195.13)
# These are the gusset "notch" vertices at x=-135.18.

# At Y midpoint, the bottom extends from (39.15, 94.49) to (-135.18, 195.13)
# Width along surface = sqrt((39.15+135.18)^2 + (94.49-195.13)^2) * cos(0)
# Actually: along the bottom surface x_dir = (0.866, 0, -0.5)
# Distance from (39.15, 94.49) to (-135.18, 195.13):
dx = -135.18 - 39.15
dz = 195.13 - 94.49
# Project onto surface x_dir: (0.866, 0, -0.5) -- this is the STEP x_dir for bottom
w_proj = abs(dx * 0.866 + dz * (-0.5))
print(f"Bottom width at midpoint (projected): {w_proj:.2f}mm")
# = |-174.33 * 0.866 + 100.64 * (-0.5)| = |-150.97 + (-50.32)| = 201.29mm
# That matches! BOTTOM_LENGTH = 201.30 ✓

# So the "full" bottom panel at the midpoint is 201.30mm wide.
# But the bottom panel shape is NOT a rectangle -- it's wider in the middle
# than at the ends because of the gussets.

# For the parametric model, we have two choices:
# A) Use a simple rectangle (width = 90.85mm at the ends)
# B) Use the full trapezoidal/pentagonal shape

# The existing code uses a simple rectangle with the 4 corner points.
# The issue is that the corners are wrong (geometry_params has wrong values).

# Let me check: what are the correct XZ corners?

# From the STEP inner faces:
# Inner BR: (35.61, 97.91) -> Outer BR: (35.61 + 1.2*(-0.5), 97.91 + 1.2*(-0.866)) = (35.01, 96.87)
# But actual outer bottom rightmost is at (39.15, 94.49) -- that's the corner chamfer extension.

# OK I think the right approach for the parametric model is:
# - Use the OUTER face corner points that appear on the END WALL profile
# - These give us the clean corners where panels meet

# From the end wall vertices at Y=-385.09:
# BR outer bottom: (39.15, 94.49) -- bottom edge of bottom panel
# Right wall bottom outer: (38.05, 99.74)
# Right wall top outer: (101.05, 208.86)
# BL outer: (-39.53, 139.91) -- left edge of bottom panel at end wall
# Gusset left: (-2.29, 204.41) -- where gusset meets end wall

# So the CORNER POINTS for the cross-section are:
# BR (where bottom meets right wall): between (39.15, 94.49) and (38.05, 99.74)
#   The actual junction is at the corner detail/chamfer.
#   The simplified intersection of the two outer surface planes is at (36.09, 96.27).
# RT (right wall top): (101.05, 208.86)
# BL (bottom left): (-39.53, 139.91) at the end walls
#   In the middle (with gusset): extends to (-135.18, 195.13)

# For the PARAMETRIC model (laser cut), we want:
# 1. Bottom: rectangle from BR to BL, where BL = (-39.53, 139.91) at ends
# 2. Right wall: rectangle from BR to RT
# 3. End walls: polygon profile

# The bottom panel rectangle width should use the END WALL profile corners.
# BR end wall: (38.05, 99.74) -- this is the right wall bottom outer vertex
# BL end wall: (-39.53, 139.91) -- bottom-left vertex on end wall

# Wait, the BR for the bottom panel and the BR for the right wall should be the SAME point
# since they share an edge. Looking at the end wall:
# (38.05, 99.74) appears as the right wall bottom outer
# (39.15, 94.49) appears as the bottom right outer
# These are two DIFFERENT vertices on the end wall! There's a gap between them
# (the corner chamfer vertices: 39.72/94.64, 41.30/97.38, 41.15/97.95).

# For the parametric model with finger joints, the bottom and right wall SHARE
# an edge at their junction. This shared edge should be at one consistent position.
# The clean intersection of the two outer planes gives us (36.09, 96.27).
# But this doesn't match either the bottom or right wall outer surface vertices.

# I think the cleanest approach is to use the RIGHT WALL outer face edges
# for the shared corner. The right wall has a clean top edge at (101.05, 208.86)
# and a bottom edge at (38.05, 99.74). Use (38.05, 99.74) as the shared corner
# between bottom and right wall.

# Then the bottom panel goes from (38.05, 99.74) to (-39.53, 139.91).
# Width = sqrt((38.05+39.53)^2 + (99.74-139.91)^2) = sqrt(77.58^2 + 40.17^2) = 87.36mm

# And the right wall goes from (38.05, 99.74) to (101.05, 208.86).
# Width = sqrt((101.05-38.05)^2 + (208.86-99.74)^2) = sqrt(63^2 + 109.12^2) = 126.00mm

rw_w = math.sqrt(63**2 + 109.12**2)
print(f"\nRight wall width (from outer vertices): {rw_w:.2f}mm")

# Compare with STEP right wall inner: 128.81mm (full inner rectangle diagonal)
# The outer right wall has many vertices due to finger joints, but the clean
# top edge at (101.05, 208.86) and bottom at (38.05, 99.74) give 126.00mm.
# The INNER right wall: from (35.61, 97.91) to (100.01, 209.46) = 128.81mm.

# The difference is because inner extends slightly further in both directions.
# The actual panel width (along the surface) is:
# Inner: sqrt((100.01-35.61)^2 + (209.46-97.91)^2) = sqrt(64.4^2 + 111.55^2) = 128.81mm
# Outer: sqrt((101.05-38.05)^2 + (208.86-99.74)^2) = sqrt(63^2 + 109.12^2) = 126.00mm

# The right wall is thinner on the outer surface due to the corner chamfers
# eating into the edges. For the parametric model, we should use the LARGER
# dimension (inner face) because that represents the full panel extent.
# The outer surface is the reference for positioning, but the panel WIDTH
# should match the inner face width.

# Actually, looking at this more carefully:
# The RIGHT WALL INNER face is a clean rectangle: 4 vertices
# (35.61, Y, 97.91) -- bottom
# (100.01, Y, 209.46) -- top
# This gives us the true panel width: 128.81mm along the surface.

# The OUTER face has finger joint notches cut into it, making it appear shorter.
# For the parametric model (before finger joints), the panel should be the FULL
# 128.81mm width.

# So the right approach: use INNER face dimensions for panel width,
# and OUTER face plane position for panel placement.

print("\n\n" + "=" * 70)
print("Right wall analysis")
print("=" * 70)

step_t = _compute_thickness_from_faces(sf_full.right_wall_inner, sf_full.right_wall_outer)
print(f"STEP thickness: {step_t:.4f}mm")

# Inner right wall corners
ri = sf_full.right_wall_inner
ri_bottom = min(ri.vertices, key=lambda v: v[2])
ri_top = max(ri.vertices, key=lambda v: v[2])
print(f"Inner bottom: ({ri_bottom[0]:.2f}, {ri_bottom[1]:.2f}, {ri_bottom[2]:.2f})")
print(f"Inner top: ({ri_top[0]:.2f}, {ri_top[1]:.2f}, {ri_top[2]:.2f})")

# Outer right wall surface position
rw_outward = _unit((0.866025, 0.0, -0.5))
outer_br = (ri_bottom[0] + step_t * rw_outward[0], 0, ri_bottom[2] + step_t * rw_outward[2])
outer_rt = (ri_top[0] + step_t * rw_outward[0], 0, ri_top[2] + step_t * rw_outward[2])
print(f"\nOuter BR (computed): ({outer_br[0]:.2f}, Y, {outer_br[2]:.2f})")
print(f"Outer RT (computed): ({outer_rt[0]:.2f}, Y, {outer_rt[2]:.2f})")

# The outer surface position should match the outer face vertices
# Outer face: (38.05, Y, 99.74) at bottom, (101.05, Y, 208.86) at top
print(f"Outer BR (from STEP): (38.05, Y, 99.74)")
print(f"Outer RT (from STEP): (101.05, Y, 208.86)")

# Computed: (35.61 + 1.2*0.866, Y, 97.91 + 1.2*(-0.5)) = (36.65, Y, 97.31)
# But STEP says (38.05, 99.74)... why?
# Because the outer face extends BEYOND the projected inner boundary.
# The outer face vertex at (38.05, 99.74) is NOT the direct offset of (35.61, 97.91).
# It's the edge of the outer face at the bottom, which includes the chamfer area.

# The key insight: the OUTER face vertices at the edges DO NOT correspond to
# the inner face vertices at those edges. The outer face is larger because of
# the chamfer geometry.

# For the parametric model, we should:
# 1. Use inner face vertices for panel dimensions (width along surface)
# 2. Compute outer surface PLANE position (d-value) from the STEP outer face
# 3. Place the panel so its outer surface is at the correct plane position

# The right wall outer PLANE: 0.866*x - 0.5*z = d
# Using any outer face vertex: 0.866*101.05 - 0.5*208.86 = 87.51 - 104.43 = -16.92
d_rw_outer = 0.866 * 101.05 - 0.5 * 208.86
print(f"\nRight wall outer plane d-value: {d_rw_outer:.4f}")

# The right wall inner PLANE: (-0.866)*x + 0.5*z = d_inner
# Using inner vertex: (-0.866)*35.61 + 0.5*97.91 = -30.84 + 48.96 = 18.12
# Wait, inner normal is (-0.866, 0, 0.5)
# Plane: -0.866*x + 0.5*z = d
# d = -0.866*35.61 + 0.5*97.91 = -30.84 + 48.96 = 18.12
d_rw_inner = -0.866 * 35.61 + 0.5 * 97.91
print(f"Right wall inner plane d-value: {d_rw_inner:.4f}")

# Distance between planes: |d_outer - (-d_inner)| / |normal| = |d_outer + d_inner|
# Wait: outer normal = (0.866, 0, -0.5), inner normal = (-0.866, 0, 0.5)
# Outer plane: 0.866*x - 0.5*z = -16.92
# Inner plane: -0.866*x + 0.5*z = 18.12 => 0.866*x - 0.5*z = -18.12
# Distance = |-16.92 - (-18.12)| = 1.20mm ✓
print(f"Thickness from plane equations: {abs(-16.92 - (-18.12)):.4f}mm")

# For the parametric model, the RIGHT WALL panel should have:
# - Outer surface at plane 0.866*x - 0.5*z = -16.92
# - Width = 128.81mm (from inner face)
# - The panel's "origin" (bottom corner on outer surface) is:
#   On the outer plane, aligned with the inner BR point (35.61, Y, 97.91)

# To find the outer origin: project inner BR onto the outer plane along the normal
# Outer_BR = inner_BR + t * outward_normal, where t is such that the result
# lies on the outer plane.
# inner_BR = (35.61, Y, 97.91)
# outward = (0.866, 0, -0.5)
# outer plane: 0.866*x - 0.5*z = -16.92
# 0.866*(35.61 + t*0.866) - 0.5*(97.91 + t*(-0.5)) = -16.92
# 30.84 + 0.75*t - 48.955 + 0.25*t = -16.92
# -18.115 + t = -16.92
# t = 1.195 ≈ 1.2mm ✓
t_rw = 1.195
outer_br_correct = (35.61 + t_rw * 0.866, 0, 97.91 + t_rw * (-0.5))
outer_rt_correct = (100.01 + t_rw * 0.866, 0, 209.46 + t_rw * (-0.5))
print(f"\nCorrect outer BR: ({outer_br_correct[0]:.2f}, Y, {outer_br_correct[2]:.2f})")
print(f"Correct outer RT: ({outer_rt_correct[0]:.2f}, Y, {outer_rt_correct[2]:.2f})")

# Good! outer BR = (36.65, Y, 97.31), outer RT = (101.05, Y, 208.86)
# The outer RT matches the STEP vertex (101.05, 208.86) ✓
# The outer BR (36.65, 97.31) does NOT match the STEP vertex (38.05, 99.74)
# because the STEP has a chamfer at the bottom corner.

# For the parametric model, we should use (36.65, 97.31) as the BR of the right wall.
# This places the outer surface correctly aligned with the inner face geometry.

# The right wall panel: from (36.65, 97.31) to (101.05, 208.86)
# Width = sqrt((101.05-36.65)^2 + (208.86-97.31)^2) = sqrt(64.4^2 + 111.55^2) = 128.81mm ✓

# Now the RIGHT WALL normal issue:
# x_dir = BR -> RT = (101.05-36.65, 0, 208.86-97.31) / 128.81 = (0.5, 0, 0.866)
# y_dir = (0, 1, 0)
# outward = x_dir x y_dir = (0.5, 0, 0.866) x (0, 1, 0) = (-0.866, 0, 0.5)
# That gives INWARD normal! We want (0.866, 0, -0.5).

# The fix: reverse x_dir to go from RT to BR instead of BR to RT.
# x_dir = RT -> BR = (36.65-101.05, 0, 97.31-208.86) / 128.81 = (-0.5, 0, -0.866)
# outward = (-0.5, 0, -0.866) x (0, 1, 0) = (0.866, 0, -0.5) ✓

# OR keep x_dir = BR -> RT but use y_dir = (0, -1, 0):
# outward = (0.5, 0, 0.866) x (0, -1, 0) = (0.866, 0, -0.5) ✓

# Let me verify: what x_dir and y_dir does the STEP outer face use?
# The STEP right wall outer: x_dir = (-0.5, 0, -0.866), y_dir = (0, 1, 0)
# outward = (-0.5, 0, -0.866) x (0, 1, 0) = (0.866, 0, -0.5) ✓

# So the fix: x_dir should go from RT to BR (down the wall), not BR to RT.

print("\n\n" + "=" * 70)
print("Bottom panel x_dir analysis")
print("=" * 70)

# Similar issue with bottom panel.
# Bottom outer normal: (-0.5, 0, -0.866)
# STEP x_dir: (0.866, 0, -0.5)
# STEP y_dir: (0, 1, 0)
# Check: (0.866, 0, -0.5) x (0, 1, 0) = (0*0 - (-0.5)*1, (-0.5)*0 - 0.866*0, 0.866*1 - 0*0)
#       = (0.5, 0, 0.866) -- that's INWARD normal, not outward!
# Hmm, the STEP x_dir gives the wrong normal sign?

# Actually in the STEP, the bottom outer face has:
# Normal: (-0.5, 0, -0.866)  [from face orientation]
# x_dir: (0.866, 0, -0.5)    [from plane axes]
# y_dir: (0, 1, 0)

# The plane's normal in OCC might be (0.5, 0, 0.866) but the face orientation
# flips it to (-0.5, 0, -0.866). The x_dir and y_dir are the PLANE's axes,
# not accounting for face orientation.

# For _make_rect_slab, we need x_dir and y_dir such that x_dir x y_dir = outward normal.
# Bottom outward: (-0.5, 0, -0.866)
# If x_dir goes from BR to BL: direction = (-0.866, 0, 0.5)
# x_dir x (0, 1, 0) = (-0.866, 0, 0.5) x (0, 1, 0)
#   = (0*0 - 0.5*1, 0.5*0 - (-0.866)*0, (-0.866)*1 - 0*0)
#   = (-0.5, 0, -0.866) ✓

# So for the bottom: x_dir = BR -> BL, y_dir = (0, 1, 0), gives outward = (-0.5, 0, -0.866) ✓
# For the right wall: x_dir = RT -> BR, y_dir = (0, 1, 0), gives outward = (0.866, 0, -0.5) ✓

# In my current code, the right wall uses x_dir = BR -> RT which gives wrong normal.
# Fix: reverse the direction.

print("\nRight wall fix needed:")
print("  Current: x_dir = BR -> RT (wrong normal)")
print("  Fix: x_dir = RT -> BR (correct outward normal)")

# Also, the bottom panel in my current code uses inner_bl which for bin_full
# is at (-38.93, Y, 140.95) -- that's the "simple" BL.
# But for variants like bin_third_left, the inner BL might be much further left
# because the bottom inner face extends to the gusset.

# Let me check bin_third_left:
print("\n\n" + "=" * 70)
print("bin_third_left: Bottom panel analysis")
print("=" * 70)

faces_tl = _extract_planar_faces('step_files/bin_third_left.step')
sf_tl = _identify_structural_faces(faces_tl)

print(f"Inner BR: ({sf_tl.inner_br[0]:.2f}, {sf_tl.inner_br[1]:.2f}, {sf_tl.inner_br[2]:.2f})")
print(f"Inner BL: ({sf_tl.inner_bl[0]:.2f}, {sf_tl.inner_bl[1]:.2f}, {sf_tl.inner_bl[2]:.2f})")

# bin_third_left inner bottom:
bi_tl = sf_tl.bottom_inner
print(f"\nBottom inner vertices ({len(bi_tl.vertices)}):")
for v in bi_tl.vertices:
    print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")

# The bottom inner for bin_third_left has 5 vertices:
# (35.61, 344.55, 97.91) -- BR
# (-136.10, 374.34, 197.05) -- far left (partition edge)
# (-150.01, 320.39, 205.08) -- gusset peak
# (-38.93, 256.25, 140.95) -- BL at front
# (35.61, 256.25, 97.91) -- BR at front

# The leftmost vertex is (-150.01, 320.39, 205.08) which is the gusset peak.
# But for the simple rectangular bottom, we want the BL at the end wall.
# For bin_third_left, the bottom extends from:
#   Front: x=-38.93 (at Y=256.25 -- front wall Y)
#   Back: x=-136.10 (at Y=374.34 -- near partition wall)

# The bottom is NOT a simple rectangle because its left edge follows the
# gusset shape. For the parametric model, we need to choose a consistent
# approach for all variants.

# APPROACH: Use the end wall profile to determine the cross-section shape,
# then the bottom panel is a rectangle with width = distance from BR to BL
# as defined by the end wall profile.

# For bin_full and bin_half_*: end walls are vertical, profile includes
# the gusset area. BL = (-39.53, 139.91) for the simple part.

# For bin_third_left: front wall is vertical, back wall is partition.
# The front wall BL is at (-39.53, 139.91).
# The partition wall extends further left.
# The bottom panel should use the front wall BL for consistency.

# Actually, for the parametric model, the bottom width should match the
# END WALL profile width. The end wall shows the full cross-section.
# The bottom edge on the end wall goes from BR to BL.

# From the front wall vertices of bin_third_left (same as bin_full):
# BL on end wall: (-39.53, 139.91)
# BR on end wall: (39.15, 94.49) -- bottom outer, or (38.05, 99.74) -- right wall bottom

# But for the partition wall (back of bin_third_left), the profile is WIDER:
# It extends from (-135.95, 196.97) on the left to (38.05, 99.74) on the right.
# The bottom edge of the partition includes the full left wall area.

# For the parametric laser-cut model, the bottom panel should be a consistent
# rectangle. The most logical width is determined by the intersection of
# bottom and right wall at BR, and the furthest-left extent of the bottom at BL.

# For ALL variants: the bottom panel's left extent is where the bottom surface
# ends at the front/back walls. For variants with vertical end walls:
# BL = (-39.53, 139.91). For variants with partition walls: the bottom
# extends further left because the partition wall shows more of the bottom.

# I think the simplest correct approach is:
# 1. BR = intersection of bottom and right wall outer planes
# 2. BL = the leftmost point on the bottom inner face's rectangular extent
# 3. Width = distance from BR to BL along the bottom surface

# Let me compute BR from plane-plane intersection:
print("\n\n" + "=" * 70)
print("Plane-plane intersection for BR corner")
print("=" * 70)

# Bottom outer: -0.5*x - 0.866*z = d_bo
# Using vertex (39.15, Y, 94.49): d_bo = -0.5*39.15 - 0.866*94.49 = -101.40
d_bo = -0.5 * 39.15 - 0.866 * 94.49
print(f"Bottom outer plane d: {d_bo:.4f}")

# Right wall outer: 0.866*x - 0.5*z = d_rw
# Using vertex (101.05, Y, 208.86): d_rw = 0.866*101.05 - 0.5*208.86 = -16.92
d_rw = 0.866 * 101.05 - 0.5 * 208.86
print(f"Right wall outer plane d: {d_rw:.4f}")

# Solve: -0.5*x - 0.866*z = d_bo and 0.866*x - 0.5*z = d_rw
# From first: x = -(d_bo + 0.866*z) / 0.5
# Sub: 0.866*(-(d_bo + 0.866*z) / 0.5) - 0.5*z = d_rw
# -1.732*(d_bo + 0.866*z) - 0.5*z = d_rw
# -1.732*d_bo - 1.5*z - 0.5*z = d_rw
# -1.732*d_bo - 2*z = d_rw
# z = (-1.732*d_bo - d_rw) / 2
z_br = (-1.732 * d_bo - d_rw) / 2
x_br = -(d_bo + 0.866 * z_br) / 0.5
print(f"\nBR corner (plane intersection): ({x_br:.4f}, {z_br:.4f})")

# Now verify: this should be consistent across all variants since they share
# the same bottom and right wall surface planes.
# Let me also compute from inner faces:
# Inner BR: (35.61, 97.91)
# Offset by step_t * outward along bottom normal:
# Bottom outward: (-0.5, 0, -0.866)
# outer_BR_bottom = (35.61 + 1.2*(-0.5), 97.91 + 1.2*(-0.866)) = (35.01, 96.87)
outer_br_b = (35.61 + 1.2 * (-0.5), 97.91 + 1.2 * (-0.866))
print(f"Outer BR from bottom offset: ({outer_br_b[0]:.2f}, {outer_br_b[1]:.2f})")

# Offset along right wall normal:
# Right outward: (0.866, 0, -0.5)
# outer_BR_right = (35.61 + 1.2*0.866, 97.91 + 1.2*(-0.5)) = (36.65, 97.31)
outer_br_r = (35.61 + 1.2 * 0.866, 97.91 + 1.2 * (-0.5))
print(f"Outer BR from right offset: ({outer_br_r[0]:.2f}, {outer_br_r[1]:.2f})")

# These two don't match because the inner BR is the CORNER where both panels meet.
# The plane intersection gives (36.09, 96.28) which is between the two.

# For the parametric model, the panels SHARE this corner.
# The bottom panel and right wall both start at the plane intersection point.

print(f"\nFor parametric model, use BR = ({x_br:.2f}, {z_br:.2f})")

# Similarly, left wall outer plane: (-0.866)*x + (-0.5)*z = d_lw
# Left wall outward: (-0.866, 0, -0.5)
# But we need to know if a left wall exists. For bin_full: no left wall.
# For the left wall, the plane equation uses the gusset face.

# Bottom BL at end wall: (-39.53, 139.91)
# Let's verify this is on the bottom outer plane:
d_check = -0.5 * (-39.53) - 0.866 * 139.91
print(f"\nBL (-39.53, 139.91) on bottom plane: d = {d_check:.4f} (expected {d_bo:.4f})")
# d = 19.765 - 121.158 = -101.393 ≈ -101.40 ✓ Yes, BL is on the bottom plane.

# For variants WITH a left wall, the BL corner is where bottom meets left wall.
# Left wall surface plane: -0.866*x - 0.5*z = d_lw
# We don't know d_lw directly, but the left wall direction is (-0.5, 0, 0.866)
# and it starts at BL. The left wall outward normal is (-0.866, 0, -0.5).
# Left wall plane: -0.866*x - 0.5*z = d_lw

# If BL is the intersection of bottom and left wall outer planes:
# Bottom: -0.5*x - 0.866*z = -101.40
# Left wall: -0.866*x - 0.5*z = d_lw
# The BL at (-39.53, 139.91) should satisfy BOTH if d_lw is correct.
# d_lw = -0.866*(-39.53) - 0.5*139.91 = 34.23 - 69.955 = -35.725
d_lw = -0.866 * (-39.53) - 0.5 * 139.91
print(f"Left wall plane d: {d_lw:.4f}")

# Verify: inner BL at (-38.93, 140.95)
# Left wall inner plane: offset by thickness inward
# Inner normal of left wall: (0.866, 0, 0.5) (opposite of outward)
# d_lw_inner = 0.866*(-38.93) + 0.5*140.95 = -33.71 + 70.48 = 36.77
# Distance = |d_lw + d_lw_inner| = |-35.73 + 36.77| = 1.04... not exactly 1.2.
# Hmm, that might be because (-39.53, 139.91) is not exactly on the left wall plane.

# Actually, the left wall doesn't exist in bin_full. The BL vertex (-39.53, 139.91)
# is where the bottom surface meets the END WALL, not a left wall.
# So the bottom panel width for bin_full is from BR to the end-wall BL.

# For bin_third_left which HAS a left wall, the BL should be at the
# bottom-left-wall intersection.

# Let me take a step back and use a simpler approach:
# For the bottom panel, use the front_wall or back_wall face to determine BL.
# Find the vertex on the end wall face with the lowest X that's also on the bottom plane.

print("\n\n" + "=" * 70)
print("End wall BL vertex analysis")
print("=" * 70)

# bin_full front wall vertices (Y=-385.09):
fw_full = sf_full.front_wall
if fw_full:
    print("bin_full front wall vertices on bottom plane (d=-101.40):")
    for v in fw_full.vertices:
        d_v = -0.5 * v[0] - 0.866 * v[2]
        if abs(d_v - d_bo) < 2.0:
            print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})  d={d_v:.4f}")

# So the vertices on/near the bottom plane are:
# (39.15, -385.09, 94.49) d=-101.40 -- BR
# (-39.53, -385.09, 139.91) d=-101.39 -- BL
# These are the two bottom-edge vertices of the end wall.

# For the parametric model:
# Bottom BL (outer) = (-39.53, 139.91)
# Bottom BR (outer) = where? The end wall has (39.15, 94.49) and (38.05, 99.74)
# but those are at slightly different d-values.

# Let me check (38.05, 99.74): d = -0.5*38.05 - 0.866*99.74 = -19.025 - 86.375 = -105.40
# That's NOT on the bottom plane (d=-101.40).

# And (39.15, 94.49): d = -0.5*39.15 - 0.866*94.49 = -19.575 - 81.828 = -101.40 ✓
# So (39.15, 94.49) IS on the bottom plane.

# And (38.05, 99.74): d = 0.866*38.05 - 0.5*99.74 = 32.95 - 49.87 = -16.92
# This IS on the right wall plane.

# Summary:
# (39.15, 94.49) is on the BOTTOM outer plane
# (38.05, 99.74) is on the RIGHT WALL outer plane
# The theoretical corner (36.09, 96.28) is on BOTH planes

# For the parametric model, the shared edge between bottom and right wall
# should be at the plane-plane intersection: (36.09, 96.28).
# The bottom panel runs from (36.09, 96.28) to (-39.53, 139.91).
# The right wall runs from (36.09, 96.28) to (101.05, 208.86).

# Width of bottom: sqrt((36.09+39.53)^2 + (96.28-139.91)^2) = sqrt(75.62^2 + 43.63^2) = 87.28mm
b_w = math.sqrt(75.62**2 + 43.63**2)
print(f"\nBottom width (plane intersection): {b_w:.2f}mm")

# Width of right wall: sqrt((101.05-36.09)^2 + (208.86-96.28)^2) = sqrt(64.96^2 + 112.58^2) = 130.0mm
rw_w2 = math.sqrt(64.96**2 + 112.58**2)
print(f"Right wall width (plane intersection): {rw_w2:.2f}mm")

# Hmm, 130.0 is different from the inner face width of 128.81.
# The difference is because the plane intersection corner is slightly different
# from the inner face corner.

# I think the most accurate approach is to use the inner face corners DIRECTLY
# for all dimension calculations, and offset the outer surface using the
# computed thickness. The inner faces are clean and precise.

# Let me compute the correct outer corners:
# Inner BR: (35.61, Y, 97.91) -- shared by bottom inner and right wall inner
# The "shared" corner on the outer surface is computed by:
# Project inner BR onto both outer planes and find the intersection.
# OR: simply offset inner BR by the STEP thickness along the BOTTOM outward normal
# for the bottom panel's origin, and offset along the RIGHT WALL outward normal
# for the right wall's origin.

# But these give DIFFERENT outer points! That's the crux of the issue.
# The bottom panel's outer BR is at (35.01, 96.87) [offset along bottom normal]
# The right wall's outer BR is at (36.65, 97.31) [offset along right wall normal]

# For the parametric model, each panel starts at its own outer surface position.
# The SHARED EDGE between them is not a single point in 3D but a line
# where both panels' surfaces meet. The finger joints connect them.

# For the shared edge definition, we need a consistent 3D line.
# The shared edge runs along Y, at a point that's on BOTH outer planes.
# That point is the plane-plane intersection: (36.09, Y, 96.28).

# For the BOTTOM panel: origin at BR = (36.09, Y_front, 96.28)
# For the RIGHT WALL: origin at BR = (36.09, Y_front, 96.28)

# This gives the correct shared edge position.
# Panel widths are then:
# Bottom: dist from (36.09, 96.28) to (-39.53, 139.91) along bottom surface
# Right wall: dist from (36.09, 96.28) to (101.05, 208.86) along right wall surface

# But wait -- this means we're not using the inner face corners at all for positioning.
# We're using the plane intersection. Let me verify that this gives consistent
# results with the inner face geometry.

# Inner BR should be on the inner surface of BOTH panels:
# Bottom inner plane: 0.5*x + 0.866*z = d_bi
# d_bi = 0.5*35.61 + 0.866*97.91 = 17.805 + 84.790 = 102.595
# Right wall inner plane: -0.866*x + 0.5*z = d_ri
# d_ri = -0.866*35.61 + 0.5*97.91 = -30.838 + 48.955 = 18.117

# If we offset the outer plane intersection (36.09, 96.28) inward:
# Along bottom inward normal (0.5, 0, 0.866): (36.09 + t*0.5, 96.28 + t*0.866)
# On bottom inner plane: 0.5*(36.09 + t*0.5) + 0.866*(96.28 + t*0.866) = 102.595
# 18.045 + 0.25*t + 83.378 + 0.75*t = 102.595
# 101.423 + t = 102.595
# t = 1.172

# Along right wall inward normal (-0.866, 0, 0.5): (36.09 + t*(-0.866), 96.28 + t*0.5)
# On right wall inner plane: -0.866*(36.09 - 0.866*t) + 0.5*(96.28 + 0.5*t) = 18.117
# -31.25 + 0.75*t + 48.14 + 0.25*t = 18.117
# 16.89 + t = 18.117
# t = 1.227

# So the inner surface of the bottom panel at the BR corner is 1.17mm from the outer.
# And the inner surface of the right wall at the BR corner is 1.23mm from the outer.
# The average is ~1.2mm, matching the STEP thickness.

# The small discrepancy (1.17 vs 1.23) is because the panels meet at 90deg,
# and the inner corner vertex (35.61, 97.91) is NOT exactly at the offset
# of the outer plane intersection.

# For the parametric model, this level of precision is more than sufficient.
# The key requirement is that the OUTER surfaces match the STEP.

print("\n\nConclusion:")
print("Use plane-plane intersection for BR corner: (36.09, 96.28)")
print("Use end-wall BL vertex for BL corner: (-39.53, 139.91)")
print("Use right wall outer top for RT corner: (101.05, 208.86)")
print("Reverse x_dir for right wall to get correct outward normal")
