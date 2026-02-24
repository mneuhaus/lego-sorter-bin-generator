"""Extract the key structural parameters from STEP files for parametric.py rewrite."""

import cadquery as cq
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
import math


def get_outer_faces(step_path):
    """Get the large structural faces (outer shell) from a STEP file."""
    shape = cq.importers.importStep(str(step_path))
    solid = shape.val()
    occ_shape = solid.wrapped

    # Overall bbox
    bbox = Bnd_Box()
    BRepBndLib.Add_s(occ_shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

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

            faces.append({
                'normal': n,
                'area': area,
                'vertices': vertices,
            })
        exp.Next()

    return faces, (xmin, ymin, zmin, xmax, ymax, zmax)


def identify_structural_faces(faces):
    """Identify the major structural panels from face data."""
    result = {}

    for f in faces:
        nx, ny, nz = f['normal']

        # Bottom outer: normal (-0.5, 0, -0.866)
        if abs(nx + 0.5) < 0.05 and abs(ny) < 0.05 and abs(nz + 0.866) < 0.05:
            if 'bottom_outer' not in result or f['area'] > result['bottom_outer']['area']:
                result['bottom_outer'] = f

        # Bottom inner: normal (0.5, 0, 0.866)
        if abs(nx - 0.5) < 0.05 and abs(ny) < 0.05 and abs(nz - 0.866) < 0.05:
            if 'bottom_inner' not in result or f['area'] > result['bottom_inner']['area']:
                result['bottom_inner'] = f

        # Right wall outer: normal (0.866, 0, -0.5)
        if abs(nx - 0.866) < 0.05 and abs(ny) < 0.05 and abs(nz + 0.5) < 0.05:
            if 'right_wall_outer' not in result or f['area'] > result['right_wall_outer']['area']:
                result['right_wall_outer'] = f

        # Right wall inner: normal (-0.866, 0, 0.5)
        if abs(nx + 0.866) < 0.05 and abs(ny) < 0.05 and abs(nz - 0.5) < 0.05:
            if 'right_wall_inner' not in result or f['area'] > result['right_wall_inner']['area']:
                result['right_wall_inner'] = f

        # Front end wall: normal (0, -1, 0)
        if abs(nx) < 0.05 and abs(ny + 1) < 0.05 and abs(nz) < 0.05:
            if 'front_wall_outer' not in result or f['area'] > result['front_wall_outer']['area']:
                result['front_wall_outer'] = f

        # Back end wall: normal (0, 1, 0)
        if abs(nx) < 0.05 and abs(ny - 1) < 0.05 and abs(nz) < 0.05:
            if 'back_wall_outer' not in result or f['area'] > result['back_wall_outer']['area']:
                result['back_wall_outer'] = f

        # Angled partition walls (for third variants)
        # Front partition: normal has -Y component (approx (-0.1287, -0.9889, 0.0743))
        if abs(abs(ny) - 0.9889) < 0.05 and abs(abs(nx) - 0.1287) < 0.05:
            if ny < 0:
                key = 'front_partition_outer'
            else:
                key = 'back_partition_outer'
            if key not in result or f['area'] > result[key]['area']:
                result[key] = f

    return result


def compute_panel_dimensions(face_data):
    """Compute the in-plane dimensions of a face from its vertices."""
    verts = face_data['vertices']
    nx, ny, nz = face_data['normal']

    # For bottom/right wall: vertices are in XYZ.
    # In-plane width = range along cross-section direction
    # In-plane height = range along Y direction

    # Project vertices to find in-plane extents
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]

    print(f"  Normal: ({nx:.4f}, {ny:.4f}, {nz:.4f})")
    print(f"  Area: {face_data['area']:.2f}")
    print(f"  X range: [{min(xs):.2f}, {max(xs):.2f}] ({max(xs)-min(xs):.2f})")
    print(f"  Y range: [{min(ys):.2f}, {max(ys):.2f}] ({max(ys)-min(ys):.2f})")
    print(f"  Z range: [{min(zs):.2f}, {max(zs):.2f}] ({max(zs)-min(zs):.2f})")
    print(f"  Vertices ({len(verts)}):")
    for v in verts:
        print(f"    ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")


# ============================================================
# bin_full analysis
# ============================================================
print("=" * 70)
print("bin_full")
print("=" * 70)

faces_full, bbox_full = get_outer_faces('step_files/bin_full.step')
print(f"BBox: X[{bbox_full[0]:.2f},{bbox_full[3]:.2f}] Y[{bbox_full[1]:.2f},{bbox_full[4]:.2f}] Z[{bbox_full[2]:.2f},{bbox_full[5]:.2f}]")
print(f"Dims: {bbox_full[3]-bbox_full[0]:.2f} x {bbox_full[4]-bbox_full[1]:.2f} x {bbox_full[5]-bbox_full[2]:.2f}")

struct_full = identify_structural_faces(faces_full)
for name, data in sorted(struct_full.items()):
    print(f"\n--- {name} ---")
    compute_panel_dimensions(data)

# Now let's compute the key corner points from the OUTER faces
# The outer faces define the outer shell. From them we extract:
# - Bottom-right corner (where bottom outer meets right wall outer)
# - Right-wall top (top of right wall outer)
# - Bottom-left corner (left end of bottom outer)

# Inner faces are offset inward by thickness (1.2mm for the STEP files)
# The inner face vertices tell us the inner corner positions

print("\n\n--- Key corner analysis from bin_full ---")
# Bottom outer face - find the extremes
bo = struct_full['bottom_outer']
bo_verts = bo['vertices']
# The rightmost vertex on bottom outer should be near the bottom-right corner
bo_xs = [v[0] for v in bo_verts]
bo_zs = [v[2] for v in bo_verts]
# Right-most bottom vertices (high X, low Z)
br_candidates = [(v[0], v[2], v[1]) for v in bo_verts if v[0] > 30]
br_candidates.sort(key=lambda v: v[0], reverse=True)
print("\nBottom outer, rightmost vertices (X, Z, Y):")
for c in br_candidates[:5]:
    print(f"  X={c[0]:.2f}, Z={c[1]:.2f}, Y={c[2]:.2f}")

# Left-most bottom vertices
bl_candidates = [(v[0], v[2], v[1]) for v in bo_verts if v[0] < -100]
bl_candidates.sort(key=lambda v: v[0])
print("\nBottom outer, leftmost vertices (X, Z, Y):")
for c in bl_candidates[:5]:
    print(f"  X={c[0]:.2f}, Z={c[1]:.2f}, Y={c[2]:.2f}")

# Right wall outer
rw = struct_full['right_wall_outer']
rw_verts = rw['vertices']
# Top of right wall = highest Z
rt_candidates = [(v[0], v[2], v[1]) for v in rw_verts]
rt_candidates.sort(key=lambda v: v[1], reverse=True)
print("\nRight wall outer, highest Z vertices:")
for c in rt_candidates[:5]:
    print(f"  X={c[0]:.2f}, Z={c[1]:.2f}, Y={c[2]:.2f}")

# Bottom of right wall
rw_bottom = [(v[0], v[2], v[1]) for v in rw_verts]
rw_bottom.sort(key=lambda v: v[1])
print("\nRight wall outer, lowest Z vertices:")
for c in rw_bottom[:5]:
    print(f"  X={c[0]:.2f}, Z={c[1]:.2f}, Y={c[2]:.2f}")


# ============================================================
# Inner face corner analysis for thickness determination
# ============================================================
print("\n\n--- Inner face analysis ---")
bi = struct_full['bottom_inner']
print("\nBottom inner vertices:")
for v in bi['vertices']:
    print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")

ri = struct_full['right_wall_inner']
print("\nRight wall inner vertices:")
for v in ri['vertices']:
    print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")

# Compute thickness from outer-to-inner face distance
# Bottom: normal = (-0.5, 0, -0.866)
# Outer vertex: (39.15, ?, 94.49)
# Inner vertex: (35.61, ?, 97.91)
# Distance = |(-0.5)*(35.61-39.15) + (-0.866)*(97.91-94.49)|
dx_bi = 35.61 - 39.15
dz_bi = 97.91 - 94.49
dist_bi = abs(-0.5 * dx_bi + -0.866 * dz_bi)
print(f"\nThickness (bottom): {dist_bi:.4f}mm")

# Right wall: normal = (0.866, 0, -0.5)
# Outer vertex: (38.05, ?, 99.74) and inner: (35.61, ?, 97.91)
dx_rw = 35.61 - 38.05
dz_rw = 97.91 - 99.74
dist_rw = abs(0.866 * dx_rw + -0.5 * dz_rw)
print(f"Thickness (right wall): {dist_rw:.4f}mm")

# ============================================================
# end wall vertex analysis
# ============================================================
print("\n\n--- End wall analysis (bin_full) ---")
if 'front_wall_outer' in struct_full:
    fw = struct_full['front_wall_outer']
    print("\nFront wall outer vertices:")
    for v in fw['vertices']:
        print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")

if 'back_wall_outer' in struct_full:
    bw = struct_full['back_wall_outer']
    print("\nBack wall outer, vertices with Y closest to ymax:")
    for v in sorted(bw['vertices'], key=lambda v: v[1], reverse=True):
        print(f"  ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")


# ============================================================
# bin_third_left
# ============================================================
print("\n\n" + "=" * 70)
print("bin_third_left")
print("=" * 70)
faces_tl, bbox_tl = get_outer_faces('step_files/bin_third_left.step')
print(f"BBox: X[{bbox_tl[0]:.2f},{bbox_tl[3]:.2f}] Y[{bbox_tl[1]:.2f},{bbox_tl[4]:.2f}] Z[{bbox_tl[2]:.2f},{bbox_tl[5]:.2f}]")

struct_tl = identify_structural_faces(faces_tl)
for name, data in sorted(struct_tl.items()):
    print(f"\n--- {name} ---")
    compute_panel_dimensions(data)


# ============================================================
# bin_third_center
# ============================================================
print("\n\n" + "=" * 70)
print("bin_third_center")
print("=" * 70)
faces_tc, bbox_tc = get_outer_faces('step_files/bin_third_center.step')
print(f"BBox: X[{bbox_tc[0]:.2f},{bbox_tc[3]:.2f}] Y[{bbox_tc[1]:.2f},{bbox_tc[4]:.2f}] Z[{bbox_tc[2]:.2f},{bbox_tc[5]:.2f}]")

struct_tc = identify_structural_faces(faces_tc)
for name, data in sorted(struct_tc.items()):
    print(f"\n--- {name} ---")
    compute_panel_dimensions(data)


# ============================================================
# bin_third_right
# ============================================================
print("\n\n" + "=" * 70)
print("bin_third_right")
print("=" * 70)
faces_tr, bbox_tr = get_outer_faces('step_files/bin_third_right.step')
print(f"BBox: X[{bbox_tr[0]:.2f},{bbox_tr[3]:.2f}] Y[{bbox_tr[1]:.2f},{bbox_tr[4]:.2f}] Z[{bbox_tr[2]:.2f},{bbox_tr[5]:.2f}]")

struct_tr = identify_structural_faces(faces_tr)
for name, data in sorted(struct_tr.items()):
    print(f"\n--- {name} ---")
    compute_panel_dimensions(data)


# ============================================================
# Compute cross-section corners from INNER faces
# ============================================================
print("\n\n" + "=" * 70)
print("Cross-section corner coordinates (from inner faces of bin_full)")
print("=" * 70)

# The inner faces form a clean rectangle (4 vertices each for right wall inner,
# and a polygon for bottom inner).
# The INNER face BR corner is at (35.61, Y, 97.91) - shared by bottom inner and right wall inner
# The INNER face RT corner is at (100.01, Y, 209.46) - top of right wall inner

# The OUTER face BR corner is at (39.15, Y, 94.49) - bottom of bottom outer
# The OUTER face RT corner is at (101.05, Y, 208.86) - top of right wall outer

# The key insight: the PARAMETRIC model should position panels using the OUTER face coordinates,
# and extrude inward. The outer face coordinates define the bin envelope.

print("\nOUTER face key corners (from bin_full):")
print(f"  Bottom-right (BR): (39.15, Y, 94.49)")
print(f"  Right-wall top (RT): (101.05, Y, 208.86)")
print(f"  Bottom-left (BL): multiple Y-ranges exist")

# Check the front wall outer face to get the full cross-section profile
if 'front_wall_outer' in struct_full:
    fw_verts = struct_full['front_wall_outer']['vertices']
    # Sort by X coordinate to understand the cross-section
    print(f"\nFront wall outer cross-section (at Y=-385.09):")
    fw_xz = sorted([(v[0], v[2]) for v in fw_verts], key=lambda p: p[0])
    for xz in fw_xz:
        print(f"  X={xz[0]:.2f}, Z={xz[1]:.2f}")


# ============================================================
# Compute Y ranges for each variant
# ============================================================
print("\n\n--- Y ranges from outer faces ---")
for variant_name, faces_data, bbox_data in [
    ("bin_full", faces_full, bbox_full),
    ("bin_third_left", faces_tl, bbox_tl),
    ("bin_third_center", faces_tc, bbox_tc),
    ("bin_third_right", faces_tr, bbox_tr),
]:
    print(f"\n{variant_name}:")
    print(f"  Y range: [{bbox_data[1]:.2f}, {bbox_data[4]:.2f}]")
    print(f"  Y extent: {bbox_data[4]-bbox_data[1]:.2f}mm")
    struct = identify_structural_faces(faces_data)
    if 'bottom_outer' in struct:
        bo_ys = [v[1] for v in struct['bottom_outer']['vertices']]
        print(f"  Bottom outer Y: [{min(bo_ys):.2f}, {max(bo_ys):.2f}]")
    if 'right_wall_outer' in struct:
        rw_ys = [v[1] for v in struct['right_wall_outer']['vertices']]
        print(f"  Right wall outer Y: [{min(rw_ys):.2f}, {max(rw_ys):.2f}]")
