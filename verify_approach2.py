"""Verify approach 2: Use inner faces (clean) + offset to derive clean outer geometry."""

import cadquery as cq
import math
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.GeomAbs import GeomAbs_Plane
from OCP.TopoDS import TopoDS
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib


def get_faces(step_path):
    shape = cq.importers.importStep(str(step_path))
    solid = shape.val()
    occ_shape = solid.wrapped
    bbox = Bnd_Box()
    BRepBndLib.Add_s(occ_shape, bbox)
    overall_bbox = bbox.Get()

    faces = []
    exp = TopExp_Explorer(occ_shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        surf = BRepAdaptor_Surface(face)
        if surf.GetType() == GeomAbs_Plane:
            plane = surf.Plane()
            normal_dir = plane.Axis().Direction()
            if face.Orientation() == TopAbs_REVERSED:
                n = (-normal_dir.X(), -normal_dir.Y(), -normal_dir.Z())
            else:
                n = (normal_dir.X(), normal_dir.Y(), normal_dir.Z())
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            area = props.Mass()
            outer_wire = BRepTools.OuterWire_s(face)
            wire_exp = BRepTools_WireExplorer(outer_wire, face)
            vertices = []
            while wire_exp.More():
                vtx = wire_exp.CurrentVertex()
                pt = BRep_Tool.Pnt_s(vtx)
                vertices.append((pt.X(), pt.Y(), pt.Z()))
                wire_exp.Next()
            faces.append({'normal': n, 'area': area, 'vertices': vertices})
        exp.Next()
    return faces, overall_bbox


def find_face(faces, normal_match, min_area=1000):
    best = None
    for f in faces:
        nx, ny, nz = f['normal']
        mx, my, mz = normal_match
        if abs(nx-mx) < 0.15 and abs(ny-my) < 0.15 and abs(nz-mz) < 0.15:
            if f['area'] >= min_area:
                if best is None or f['area'] > best['area']:
                    best = f
    return best


# ============================================================
# Analysis from bin_full inner faces
# ============================================================
print("=" * 70)
print("Key geometry from bin_full INNER faces")
print("=" * 70)

faces, overall_bbox = get_faces('step_files/bin_full.step')

# The INNER faces are clean (no finger joints):
# Bottom inner: normal (0.5, 0, 0.866), 4 main corner vertices + gusset vertices
# Right wall inner: normal (-0.866, 0, 0.5), clean rectangle

bottom_inner = find_face(faces, (0.5, 0, 0.866))
right_inner = find_face(faces, (-0.866, 0, 0.5))

print(f"\nBottom inner (area={bottom_inner['area']:.2f}):")
bi_verts = bottom_inner['vertices']
for v in bi_verts:
    print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")

print(f"\nRight wall inner (area={right_inner['area']:.2f}):")
ri_verts = right_inner['vertices']
for v in ri_verts:
    print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")

# The inner right wall is a clean rectangle:
# (35.61, -82.73, 97.91)  -- BR inner
# (35.61, -383.75, 97.91)  -- BR inner (other end)
# (100.01, -383.75, 209.46) -- RT inner (other end)
# (100.01, -82.73, 209.46)  -- RT inner

# STEP thickness = 1.2mm
# Outer = inner + 1.2 * outward_normal

STEP_T = 1.2

# Bottom outer normal = (-0.5, 0, -0.866)
# Inner BR at (35.61, Y, 97.91) -> Outer BR = (35.61 + 1.2*(-0.5), Y, 97.91 + 1.2*(-0.866))
#                                             = (35.01, Y, 96.87)
# Wait, but the actual outer vertices at the bottom-right are at (39.15, Y, 94.49)
# That doesn't match! The inner BR is at (35.61, Y, 97.91).
# Let me check: (35.61 - 1.2*0.5, Y, 97.91 - 1.2*0.866) = (35.01, Y, 96.87)
# But the outer face bottom-right is at (39.15, Y, 94.49) -- much further!

# The discrepancy is because the inner face's BR is where BOTTOM meets RIGHT WALL,
# but the outer face extends further (the chamfer / corner detail at the junction).

# Let me look at this more carefully.
# Inner bottom: (35.61, Y, 97.91) - shared with right wall inner
# Inner right wall: (35.61, Y, 97.91) - shared with bottom inner at bottom
# Inner right wall top: (100.01, Y, 209.46)

# The outer bottom-right edge is NOT a simple offset of the inner.
# The corner has small chamfer edges (the corner detail).

# So the correct approach: look at the outer END WALL face to get the clean
# cross-section profile, since the end wall sits at a single Y plane and
# shows the complete cross-section.

print("\n\n--- End wall cross-section analysis ---")
front_outer = find_face(faces, (0, -1, 0))
print(f"\nFront wall outer vertices (at Y=-385.09):")
fw_verts = sorted(front_outer['vertices'], key=lambda v: v[0])
for v in fw_verts:
    print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")

# The front wall profile in XZ is:
# (-39.53, 139.91) -- bottom-left (where bottom meets left end)
# (-2.29, 204.41)  -- left side: gusset meets something
# (-0.97, 203.66)  -- corner detail
# (1.13, 203.09)   -- ledge left
# (37.10, 203.09)  -- ledge right
# (38.05, 99.74)   -- right wall bottom (OUTER)
# (39.15, 94.49)   -- bottom-right corner (very bottom of outer bottom)
# (39.72, 94.64)   -- corner detail
# (41.15, 97.95)   -- corner detail
# (41.30, 97.38)   -- corner detail
# (53.48, 231.47)  -- ramp top
# (59.22, 233.01)  -- right wall ledge top
# (101.05, 208.86) -- right wall top

# The inner end wall would give us the clean profile without outer corner details.
# But actually, the outer end wall IS the clean profile for the end wall shape.
# The notches/corner details ARE part of the end wall shape.

# For the PARAMETRIC model, we want:
# 1. Bottom panel: a clean rectangle between (BR corner, BL corner) x (front, back)
# 2. Right wall: a clean rectangle between (BR corner, RT corner) x (front, back)
# 3. End walls: the polygon profile from the end wall face

# The "clean" corners for the rectangular panels should be:
# - Where bottom line meets right wall line (the intersection of the two planes)
# - Where bottom line meets left wall line

# Bottom plane direction: from (39.15, Y, 94.49) toward (-39.53, Y, 139.91)
# That's the direction of the bottom outer edge.
# But this isn't quite right because (39.15, 94.49) to (-39.53, 139.91) gives us
# a vector pointing from BR to BL.

# Let me compute the actual line-line intersection.
# Bottom surface: passes through (39.15, Y, 94.49) with normal (-0.5, 0, -0.866)
# Right wall surface: passes through (101.05, Y, 208.86) with normal (0.866, 0, -0.5)

# The intersection of these two planes is a line along Y.
# Point on both planes (at some Y):
# Bottom: -0.5*x - 0.866*z = -0.5*39.15 - 0.866*94.49 = -19.575 - 81.828 = -101.403
# Right:   0.866*x - 0.5*z = 0.866*101.05 - 0.5*208.86 = 87.509 - 104.43 = -16.921
#
# System: -0.5*x - 0.866*z = -101.403
#          0.866*x - 0.5*z = -16.921
#
# From eq 1: x = (-101.403 + 0.866*z) / (-0.5) = 202.806 - 1.732*z
# Sub into eq 2: 0.866*(202.806 - 1.732*z) - 0.5*z = -16.921
#                175.63 - 1.4999*z - 0.5*z = -16.921
#                175.63 - 2.0*z = -16.921
#                2.0*z = 192.551
#                z = 96.276
#                x = 202.806 - 1.732*96.276 = 202.806 - 166.711 = 36.095

print(f"\nPlane-plane intersection (bottom x right wall):")
print(f"  BR corner: x=36.10, z=96.28")
print(f"  (Compare: inner face BR at x=35.61, z=97.91)")
print(f"  (Compare: outer face right wall bottom at x=38.05, z=99.74)")
print(f"  (Compare: outer face bottom right at x=39.15, z=94.49)")

# Hmm, the intersection is at (36.10, 96.28), which is between the inner and outer values.
# This makes sense - the intersection of the outer surface planes doesn't match any specific vertex
# because the corner has a chamfer.

# Let me instead look at what the inner face junction actually is.
# Inner bottom BR at (35.61, Y, 97.91)
# Inner right wall BR at (35.61, Y, 97.91) -- same point!
# So the inner faces meet exactly at (35.61, Y, 97.91).

# Inner right wall top at (100.01, Y, 209.46)
# Inner bottom left: look at the bottom inner face:
#   (-38.93, Y, 140.95) -- the simple BL corner (front/back ends)
#   (-150.01, Y, 205.08) -- the gusset points
#   (-134.58, Y, 196.17) -- gusset inner vertex

# So the inner rectangle for the bottom panel is bounded by:
#   BR: (35.61, Y, 97.91)
#   BL: (-38.93, Y, 140.95) for the front/back "short" side
#   BL: (-150.01, Y, 205.08) for the "long" side (gusset extension)

# The bottom inner face is NOT a rectangle - it has gusset indentations.
# But for the "clean" rectangular base, the main corners are:
#   BR: (35.61, Y, 97.91) and BL: (-38.93, Y, 140.95)

# Now let's compute outer positions from inner positions:
# Outer = Inner + STEP_T * outward_normal
# For bottom: outward = (-0.5, 0, -0.866)
# Inner BR: (35.61, Y, 97.91) -> Outer BR: (35.61 + 1.2*(-0.5), Y, 97.91 + 1.2*(-0.866))
#                                          = (35.01, Y, 96.87)
# But actual outer face bottom edge is at x=39.15, z=94.49... that's quite different.

# Wait - the outer face vertices are NOT the direct offset of the inner face.
# The outer face has a larger footprint because of the corner chamfer at the
# bottom-right junction. The outer bottom face extends from x=-151.11 to x=39.15,
# while the inner face extends from x=-150.01 to x=35.61.

# The outer face is offset OUTWARD from the inner face by thickness=1.2 in the
# normal direction. Let me verify:
# Inner face point: (35.61, -82.73, 97.91), normal (0.5, 0, 0.866) (inward)
# Outward normal for bottom: (-0.5, 0, -0.866)
# Outer surface: (35.61 + 1.2*(-0.5), -82.73, 97.91 + 1.2*(-0.866)) = (35.01, -82.73, 96.87)
# But the outer face vertex nearest this is (39.15, -81.39, 94.49)

# The difference is: the bottom outer face EXTENDS PAST the inner face boundary
# at the right side because the corner chamfer adds extra material there.
# The outer face actually extends further right (x=39.15 vs x=35.61) because
# of the corner detail between bottom and right wall.

# So the actual "envelope corner" (where bottom and right wall outer surfaces intersect)
# is at the computed (36.10, z=96.28) -- this is the theoretical intersection.
# The actual geometry has a small chamfer (3.5mm edges) at this junction.

# For the PARAMETRIC model, we should use the clean rectangular envelope.
# The key question: what are the correct envelope corners?

# Let me use a different approach: compute from the end wall profile.
# The end wall shows the full cross-section including chamfer details.
# For the parametric model, we want the simplified version (no chamfer).

# Looking at the end wall profile sorted by X:
# (-39.53, 139.91) -- BL outer (the furthest-left "simple" point on bottom)
# (-2.29, 204.41)  -- where gusset meets end wall
# (-0.97, 203.66)  -- gusset-to-ledge transition
# (1.13, 203.09)   -- ledge left
# (37.10, 203.09)  -- ledge right
# (38.05, 99.74)   -- right wall bottom outer
# (39.15, 94.49)   -- bottom right outer corner
# (39.72, 94.64)   -- chamfer vertex
# (41.15, 97.95)   -- chamfer vertex (top of chamfer)
# (41.30, 97.38)   -- chamfer vertex
# (53.48, 231.47)  -- ledge ramp top
# (59.22, 233.01)  -- ledge ramp top right
# (101.05, 208.86) -- right wall top outer

# For the simplified parametric model, the key cross-section corners are:
# BR (bottom-right): where bottom outer surface meets right wall outer surface = (36.10, 96.28)
# RT (right wall top): (101.05, 208.86) -- this is a clean corner
# BL (bottom-left): (-39.53, 139.91) at the front/back ends
#                    (-151.11, 204.33) at the gusset peak

# The RIGHT WALL inner face goes from (35.61, 97.91) to (100.01, 209.46)
# The width of the right wall inner = sqrt((100.01-35.61)^2 + (209.46-97.91)^2)
rw_w = math.sqrt((100.01-35.61)**2 + (209.46-97.91)**2)
print(f"\nRight wall inner width: {rw_w:.2f}mm")

# The bottom inner face, clean part from (-38.93, 140.95) to (35.61, 97.91)
b_w = math.sqrt((35.61-(-38.93))**2 + (97.91-140.95)**2)
print(f"Bottom inner width (clean, no gusset): {b_w:.2f}mm")

# Now let me look at this from a different angle.
# The STEP files already define the EXACT geometry we need.
# For the parametric model, we should:
# 1. Use the outer face vertices directly for the clean corners
# 2. Accept that there's a small chamfer at BR junction
# 3. For the simplified rectangular panels, use the INNER face rectangle
#    and offset outward
#
# Or even simpler: for the rectangular panels (bottom, right wall, left wall),
# use the INNER face vertices directly for the inner surface,
# and just set the thickness to grow outward from there.
# This gives us EXACTLY matching inner geometry, and outer geometry
# that might differ slightly from the original STEP at the corners.
#
# But the task says "outer surfaces should match the STEP exactly".
# So we need to preserve the outer surface positions.

# The approach: use the right wall and bottom outer surface PLANE positions
# (derived from the d-value of each plane equation), and compute the
# rectangular panel with the correct Y extents.

# Right wall outer plane: 0.866*x - 0.5*z = d_rw
# Using vertex (101.05, Y, 208.86): d_rw = 0.866*101.05 - 0.5*208.86 = -16.92
# Using vertex (38.05, Y, 99.74): d_rw = 0.866*38.05 - 0.5*99.74 = 32.95 - 49.87 = -16.92 ✓

# Bottom outer plane: -0.5*x - 0.866*z = d_bo
# Using vertex (39.15, Y, 94.49): d_bo = -0.5*39.15 - 0.866*94.49 = -19.575 - 81.828 = -101.40
# Using vertex (-39.53, Y, 139.91): d_bo = -0.5*(-39.53) - 0.866*139.91 = 19.765 - 121.16 = -101.40 ✓

# The planes are consistent. The corners where the bottom and right wall outer
# surfaces intersect are:
# -0.5*x - 0.866*z = -101.40  (bottom)
# 0.866*x - 0.5*z = -16.92    (right wall)
#
# Solving:
# x = (101.40 - 0.866*z) / 0.5 = 202.80 - 1.732*z
# 0.866*(202.80 - 1.732*z) - 0.5*z = -16.92
# 175.62 - 1.5*z - 0.5*z = -16.92
# -2*z = -192.54
# z = 96.27
# x = 202.80 - 1.732*96.27 = 202.80 - 166.71 = 36.09

print(f"\nPlane intersection corner (bottom x right): ({36.09:.2f}, {96.27:.2f})")

# Now let's use these exact plane positions for the parametric model.
# The idea: the OUTER plane position is canonical. For the parametric model:
# - Bottom panel: outer surface at the computed plane position, grows inward
# - Right wall: outer surface at the computed plane position, grows inward

# For a CLEAN rectangular panel, the corners in the XZ cross-section are:
# BR: intersection of bottom and right wall planes = (36.09, 96.27)
# RT: top of right wall. The right wall outer face top is at (101.05, 208.86).
# BL: bottom-left. The bottom outer face left edge goes to (-39.53, 139.91)
#     at the front/back, or (-151.11, 204.33) at the gusset peak.

# For the simple (no-gusset) case:
# BL: (-39.53, 139.91) -- but this is just the front/back end wall vertex.
# The true BL in the mid-section depends on whether there's a left wall.

# Wait, I realize the issue: the bottom panel is NOT rectangular in the XZ
# cross-section for bin_full. It has gussets on the left side. The "clean"
# bottom runs from BR to BL in XZ, where BL is at (-39.53, 139.91).
# BUT in the mid-section (away from front/back), the bottom extends further
# left because of the gusset triangles (to -151.11, 204.33).

# For the parametric model, we treat the bottom as a simple rectangle:
# width = distance from BR to BL in the bottom plane
# height = Y depth

# The bottom panel for bin_full goes from the front Y to back Y:
# Front outer: Y = -385.09
# Back outer: Y = -81.39

# For the bottom panel, the CLEAN rectangle has:
# Right edge at x=39.15, z=94.49 (from outer face)
# Left edge at x=-39.53, z=139.91 (from outer face - at front/back end)
# No wait, those are the end wall vertices. The actual bottom panel extends
# further left in the middle (gusset area).

# Hmm, this is getting complex. Let me take a completely different approach.

# APPROACH 3: Use the END WALL (front/back) face profile as the cross-section shape.
# The end wall defines the EXACT polygon. For rectangular panels (bottom, right wall),
# we extract the relevant edge segment from the end wall profile.

# The front wall outer face at Y=-385.09 has these vertices (sorted by X):
# (-39.53, 139.91) -- BL
# (-2.29, 204.41) -- gusset corner
# ... etc ...
# (101.05, 208.86) -- RT

# The BOTTOM panel's right edge starts at x=39.15, z=94.49 (or more precisely
# at the corner detail: 38.05/99.74 to 41.30/97.38).
# The BOTTOM panel's left edge is at x=-39.53, z=139.91.

# So the bottom edge (left-to-right) of the end wall cross-section is:
# From (-39.53, 139.91) to (39.15, 94.49) -- this IS the bottom panel edge.

# Let me compute the width of this edge:
bw = math.sqrt((-39.53-39.15)**2 + (139.91-94.49)**2)
print(f"\nBottom edge width (front wall): {bw:.2f}mm")
# Compare with bottom outer face in-plane width
# Bottom outer normal: (-0.5, 0, -0.866), x_dir: (0.866, 0, -0.5)
# From vertex (39.15, Y, 94.49) to (-39.53, Y, 139.91):
dx_b = -39.53 - 39.15
dz_b = 139.91 - 94.49
w_along_xdir = dx_b * 0.866 + dz_b * (-0.5)
print(f"Bottom width along surface x_dir: {abs(w_along_xdir):.2f}mm")

# The right wall edge (bottom-to-top) from the end wall:
# From (38.05, 99.74) to (101.05, 208.86)
dx_r = 101.05 - 38.05
dz_r = 208.86 - 99.74
rw_edge = math.sqrt(dx_r**2 + dz_r**2)
print(f"Right wall edge (from end wall): {rw_edge:.2f}mm")

# Alternative: from BR intersection (36.09, 96.27) to (101.05, 208.86)
dx_r2 = 101.05 - 36.09
dz_r2 = 208.86 - 96.27
rw_edge2 = math.sqrt(dx_r2**2 + dz_r2**2)
print(f"Right wall edge (from plane intersection): {rw_edge2:.2f}mm")


# OK let me just test the simplest possible approach:
# Extract the EXACT outer face outline (including finger joint notches),
# use it to create the panel solid. This should give an EXACT match.
# Then for the parametric model, we'll use a SIMPLIFIED version of the outline.

# Actually, the task description says to match the OUTER SURFACES.
# The simplest way is to extract the outer face polygon and use it directly.
# The finger joint notches on the outer face of the STEP are part of the
# original design -- we want our parametric model to produce CLEAN panels
# (without finger joints -- joints are added later by joints.py).

# So I need the CLEAN outer face outline, not the one with finger joints.
# The CLEAN outline = the outer face outline with finger joint notches removed.

# For the bottom panel, the clean outline is a rectangle:
#   BR corner at (39.15, Y_front, 94.49) to (39.15, Y_back, 94.49) -- right edge
#   BL corner at (-39.53, Y_front, 139.91) to (-39.53, Y_back, 139.91) -- left edge
#   But wait, the left side has gusset geometry (it's not at a single X position).

# The bottom outer face vertices that define the LEFT edge:
# (-39.53, -385.09, 139.91) -- front-left
# (-151.11, -320.67, 204.33) -- gusset peak (back half)
# ... gusset vertices ...
# (-151.11, -145.81, 204.33) -- gusset peak (front half)
# (-39.53, -81.39, 139.91) -- back-left

# So the bottom panel is NOT a rectangle! It has two triangular gusset
# extensions on the left side. The "base" rectangle goes from
# (-39.53, Y, 139.91) to (39.15, Y, 94.49), but there are additional
# triangular regions extending left to (-151.11, Y, 204.33).

# For the PARAMETRIC model, we need to handle:
# 1. bin_full: bottom has gussets, no left wall panel
# 2. bin_third_left: bottom has one gusset at the partition end
# 3. bin_third_center: bottom has no gussets (just rectangle to partition walls)
# 4. bin_third_right: bottom has one gusset at the partition end

# This is complex. Let me look at what the existing code does vs what it should do.

# EXISTING CODE uses _corners() which gives 4 XZ corner points.
# The issue is these corners don't match the STEP geometry.
# The fix: extract correct corner positions and Y extents from each STEP variant.

# Let me now write the CORRECT extraction for each variant.

# For bin_full:
# - Bottom: from (-39.53, 139.91) to (39.15, 94.49) in XZ, extended left with gussets
# - Right wall: from (38.05, 99.74) to (101.05, 208.86) in XZ
#   (but 38.05 is the "inner edge" of the right wall outer face, including corner detail)
# - Left wall: NO full left wall for bin_full (only gussets)
# - Front wall: polygon at Y=-385.09
# - Back wall: polygon at Y=-81.39

# The STEP thickness is 1.2mm. Inner face positions:
# Bottom inner BR: (35.61, Y, 97.91)
# Right wall inner BR: (35.61, Y, 97.91)
# Right wall inner RT: (100.01, Y, 209.46)
# Bottom inner BL: (-38.93, Y, 140.95)

# Thickness in perpendicular direction:
# Bottom: inner to outer distance along normal (-0.5, 0, -0.866)
# From inner (35.61, Y, 97.91) to outer (39.15, Y, 94.49):
# offset = (39.15-35.61, 0, 94.49-97.91) = (3.54, 0, -3.42)
# dot with outward normal: 3.54*(-0.5) + (-3.42)*(-0.866) = -1.77 + 2.96 = 1.19 ≈ 1.2mm ✓

# But the outer face EXTENDS BEYOND the inner face boundary:
# Inner bottom BR is at (35.61, 97.91), but outer bottom goes to (39.15, 94.49)
# That's because the outer face includes the bottom side of the corner chamfer.

# For the parametric model with a DIFFERENT thickness:
# We should keep the outer surface at the same position, but the inner
# surface will be at a different position.

# The key constraint is: "outer surfaces should match the STEP exactly"
# This means: the outer face of each panel should be at the same plane
# position as in the STEP.

# For a rectangular panel (bottom, right wall, left wall):
# - Outer surface plane position is fixed (from STEP)
# - Panel width along the surface = same as STEP
# - Panel height (Y extent) = same as STEP variant
# - Thickness grows inward from the outer surface

# For the bottom panel in bin_full:
# The "rectangular" part runs from BR to BL, but the full shape includes gussets.
# Since gussets are separate triangle faces in the STEP, we DON'T include them
# in the bottom panel. The bottom panel is just the main rectangular part.

# Wait, but the bottom OUTER face in the STEP IS a single face that includes
# both the main rectangle AND the gusset extensions. They're one face because
# they're coplanar.

# Hmm, I need to reconsider. The gusset triangular regions are part of the
# bottom surface. They exist as the triangular "fill" between the bottom
# panel and the left wall gusset.

# For the parametric model, the bottom panel should be the full outer surface
# footprint (including gusset extensions). This means the bottom panel is
# NOT a simple rectangle.

# Actually wait - for laser cutting, the bottom panel IS cut as a rectangle
# (or maybe a polygon with the gusset tabs). The gusset regions would need
# to be separate pieces that fold up to form the left wall.

# Let me re-read the architecture:
# The model has 5 panels: bottom, right_wall, left_wall, front_wall, back_wall
# The left wall in bin_full is a gusset (triangular, partial depth)
# The bottom panel is the main trough floor

# I think the right approach is:
# 1. Bottom panel = rectangle from BR to BL, Y_front to Y_back
# 2. Right wall = rectangle from BR to RT, Y_front to Y_back
# 3. Left wall (if full) = rectangle from BL to LT, Y_front to Y_back
# 4. Front/back walls = polygon profile

# The gusset triangular faces are NOT separate panels - they're part of the
# 3D printed/molded original. For laser cutting, we approximate with the
# panels we can make.

# For the bounding box match, we need:
# - The bottom + right wall + left wall + end walls together should produce
#   the correct overall bounding box.

# The STEP bbox for bin_full is:
# X: [-155.73, 101.05] (256.78)
# Y: [-385.09, -81.38] (303.71)
# Z: [94.48, 267.35] (172.87)
#
# Our panels should reach these extremes:
# X min: around -151 from bottom BL (-151.11 from gusset peak)
#   But our bottom rectangle only goes to -39.53... that leaves ~115mm missing
#   The gusset extends the X range. Without gusset panels, we can't match X min.

# OK I think I understand now. The task says the bounding box is:
# X: [-151.11, 101.05] (252.16mm)
# Y: [-385.09, -81.39] (303.70mm)
# Z: [94.49, 265.92] (171.43mm)

# Note: these are slightly different from what I measured (the STEP bbox includes
# the full solid including inner surfaces and chamfers).

# The "252.16 x 303.70 x 171.43" target in the task description comes from
# geometry_params.py which has bbox_x=252.16 and bbox_z=171.43.

# Now, (-151.11) to (101.05) = 252.16  ✓
# But the STEP bbox X range is [-155.73, 101.05] = 256.78, not 252.16.
# The extra X extent comes from the gusset inner faces and small details.

# So the TARGET bbox (252.16 x 303.70 x 171.43) represents the OUTER SHELL
# bbox, not the full solid bbox. That makes sense.

# The OUTER shell faces extend:
# X: [-151.11 (bottom outer left), 101.05 (right wall outer top)]
# Y: [-385.09, -81.39]
# Z: [94.49 (bottom outer right), ???]

# Z max: what's the highest Z on the outer shell?
# Right wall outer top: Z=208.86
# End wall top: Z=233.01
# Gusset face vertices go up to Z=265.97 (from the triangular gusset faces)
# The gusset outer face normals are (-0.387, +-0.894, 0.224)

# So the bbox_z=171.43 means Z max = 94.49 + 171.43 = 265.92
# Which matches the gusset face vertices (265.97 ≈ 265.92)

# This means the FULL outer shell bounding box includes the gusset geometry.
# The parametric model needs to include gusset faces or left wall faces
# to reach the correct Z extent.

# For bin_full with a full left wall (bin_third_left): the left wall top
# reaches the correct Z.

# For bin_full WITHOUT a full left wall: the gusset triangular faces
# provide the Z extent. These are separate from the 5 main panels.

# The task says to match 252.16 x 303.70 x 171.43. But with only
# bottom + right wall + front/back walls, the Z max is only 233.01
# (from end wall top). We need 265.92.

# This means we MUST include the left wall gusset panels in the model
# for the bbox to match.

# For bin_full, the "left_wall" panel in the STEP is not a single rectangle
# but consists of triangular gusset faces at front and back.

# Hmm wait, bin_full has has_left_wall_full=False but has_left_wall_gusset=True.
# The existing code only creates a left wall if has_left_wall_full=True.
# So for bin_full, there's no left wall panel, and the bbox WON'T match in Z.

# Let me check: what's the Z extent without the gusset?
# Right wall outer top: Z=208.86
# End wall top: Z=233.01 (from the ledge ramp vertex at (59.22, Y, 233.01))
# So Z range = [94.49, 233.01] = 138.52mm (not 171.43mm)

# The missing 33mm comes from the gusset. For the parametric model to match
# the STEP bbox, we'd need to add gusset panels.

# BUT the task says to focus on the 5 structural panels and get them right.
# Let me check what the variant-specific bbox should be:
# bin_full: bbox_z=171.43 (with gusset)
# bin_third_left: bbox_z=171.43 (with full left wall)
# bin_third_center: bbox_z=165.95 (no left wall)
# bin_third_right: bbox_z=171.43 (with full left wall)

# For bin_third_center, bbox_z=165.95, Z range = [94.49, 260.44]
# The 260.44 comes from the partition wall vertex. So the Z extent is
# determined by the partition walls, not gussets.

# For bin_third_left: the full left wall top is high enough for 171.43.

# I think the right approach is:
# 1. For variants with full left wall (third_left, third_right): include left wall panel
# 2. For bin_full: the bbox target might need gusset support panels
# 3. For bin_third_center: partition walls provide the Z extent

# Let me check if maybe the existing code just doesn't match for bin_full's Z
# and that's OK because the "5 panels" for bin_full are:
# bottom, right_wall, front_wall, back_wall (only 4 panels, no left wall)

# Re-reading the task: it says "5 structural panels" but bin_full only has 4.
# The task says "verify bin_third_left (5 panels)".
# So bin_full has 4 panels, bin_third_left has 5.

# Let me just focus on getting the outer face positions RIGHT for each panel.
# The bounding box will match as closely as the panels allow.

print("\n\n" + "=" * 70)
print("FINAL: Key dimensions to use for parametric model")
print("=" * 70)

# From the front wall outer face vertices, the cross-section profile is:
print("\nEnd wall cross-section profile (XZ, from front wall outer of bin_full):")
profile_xz = [
    (38.05, 99.74),     # right wall bottom outer
    (101.05, 208.86),   # right wall top outer
    (59.22, 233.01),    # ledge top right
    (53.48, 231.47),    # ledge top left
    (37.10, 203.09),    # ledge bottom right
    (1.13, 203.09),     # ledge bottom left
    (-0.97, 203.66),    # transition
    (-2.29, 204.41),    # gusset corner
    (-39.53, 139.91),   # bottom-left outer
    (39.15, 94.49),     # bottom-right outer (lowest point)
    (39.72, 94.64),     # chamfer vertex 1
    (41.30, 97.38),     # chamfer vertex 2
    (41.15, 97.95),     # chamfer vertex 3
]
for p in profile_xz:
    print(f"  ({p[0]:.2f}, {p[1]:.2f})")

# Key Y extents for each variant
print("\nY extents per variant:")
for name, step_file in [
    ("bin_full", "step_files/bin_full.step"),
    ("bin_third_left", "step_files/bin_third_left.step"),
    ("bin_third_center", "step_files/bin_third_center.step"),
    ("bin_third_right", "step_files/bin_third_right.step"),
    ("bin_half_left", "step_files/bin_half_left.step"),
    ("bin_half_right", "step_files/bin_half_right.step"),
]:
    fcs, bb = get_faces(step_file)
    # Get Y from the largest bottom outer face
    bo = find_face(fcs, (-0.5, 0, -0.866))
    rw = find_face(fcs, (0.866, 0, -0.5))
    if bo and rw:
        bo_ys = [v[1] for v in bo['vertices']]
        rw_ys = [v[1] for v in rw['vertices']]
        y_front = min(min(bo_ys), min(rw_ys))
        y_back = max(max(bo_ys), max(rw_ys))
        print(f"  {name}: Y_front={y_front:.2f}, Y_back={y_back:.2f}, depth={y_back-y_front:.2f}")

        # Also check for partition walls
        front_part = find_face(fcs, (-0.1287, -0.9889, 0.0743))
        back_part = find_face(fcs, (-0.1287, 0.9889, 0.0743))
        front_part2 = find_face(fcs, (0.1287, -0.9889, -0.0743))
        back_part2 = find_face(fcs, (0.1287, 0.9889, -0.0743))
        if front_part:
            print(f"    Has front partition: normal=({front_part['normal'][0]:.4f},{front_part['normal'][1]:.4f},{front_part['normal'][2]:.4f}) area={front_part['area']:.0f}")
        if front_part2:
            print(f"    Has front partition2: normal=({front_part2['normal'][0]:.4f},{front_part2['normal'][1]:.4f},{front_part2['normal'][2]:.4f}) area={front_part2['area']:.0f}")
        if back_part:
            print(f"    Has back partition: normal=({back_part['normal'][0]:.4f},{back_part['normal'][1]:.4f},{back_part['normal'][2]:.4f}) area={back_part['area']:.0f}")
        if back_part2:
            print(f"    Has back partition2: normal=({back_part2['normal'][0]:.4f},{back_part2['normal'][1]:.4f},{back_part2['normal'][2]:.4f}) area={back_part2['area']:.0f}")

# Check for left wall faces
print("\nLeft wall presence per variant:")
for name, step_file in [
    ("bin_full", "step_files/bin_full.step"),
    ("bin_third_left", "step_files/bin_third_left.step"),
    ("bin_third_center", "step_files/bin_third_center.step"),
    ("bin_third_right", "step_files/bin_third_right.step"),
]:
    fcs, _ = get_faces(step_file)
    # Left wall outer: normal (-0.866, 0, -0.5) or similar
    # Actually from the data, the left wall faces have different normals
    # depending on the variant. Let me check all large faces.
    print(f"\n  {name} - faces with area > 5000:")
    for f in sorted(fcs, key=lambda x: x['area'], reverse=True):
        if f['area'] > 5000:
            nx, ny, nz = f['normal']
            print(f"    n=({nx:.4f},{ny:.4f},{nz:.4f}) area={f['area']:.0f}")
