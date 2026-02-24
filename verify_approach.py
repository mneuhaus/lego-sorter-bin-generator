"""Verify the STEP extraction approach before rewriting parametric.py.

This script extracts outer face data from STEP files, builds simplified panel
solids from those faces, and compares bounding boxes.
"""

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


def get_face_data(step_path):
    """Extract all planar face data from a STEP file."""
    shape = cq.importers.importStep(str(step_path))
    solid = shape.val()
    occ_shape = solid.wrapped

    # Overall bounding box
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
            loc = plane.Location()
            ax = plane.Position()
            x_dir = ax.XDirection()
            y_dir = ax.YDirection()

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
                'plane_origin': (loc.X(), loc.Y(), loc.Z()),
                'x_dir': (x_dir.X(), x_dir.Y(), x_dir.Z()),
                'y_dir': (y_dir.X(), y_dir.Y(), y_dir.Z()),
            })
        exp.Next()

    return faces, overall_bbox


def find_outer_face(faces, normal_match, min_area=1000):
    """Find the largest face matching a given normal direction."""
    best = None
    for f in faces:
        nx, ny, nz = f['normal']
        mx, my, mz = normal_match
        if (abs(nx - mx) < 0.05 and abs(ny - my) < 0.05 and abs(nz - mz) < 0.05):
            if f['area'] >= min_area:
                if best is None or f['area'] > best['area']:
                    best = f
    return best


def compute_face_y_range(face):
    """Get the Y range of a face's vertices."""
    ys = [v[1] for v in face['vertices']]
    return min(ys), max(ys)


def project_to_plane(vertices, normal, plane_origin):
    """Project 3D vertices onto a 2D plane defined by normal and origin.

    Returns (u_dir, v_dir, uv_points) where uv_points are the 2D coordinates.
    """
    nx, ny, nz = normal

    # Choose appropriate u and v axes
    if abs(ny) > 0.9:
        # Mostly Y normal (end wall / partition)
        # u = world X, v = world Z (approximately)
        u_candidate = (1, 0, 0)
    elif abs(nx) > abs(nz):
        # Mostly X normal
        u_candidate = (0, 1, 0)
    else:
        u_candidate = (1, 0, 0)

    # Gram-Schmidt to get u orthogonal to normal
    dot_un = u_candidate[0]*nx + u_candidate[1]*ny + u_candidate[2]*nz
    u = (
        u_candidate[0] - dot_un * nx,
        u_candidate[1] - dot_un * ny,
        u_candidate[2] - dot_un * nz,
    )
    u_mag = math.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    u = (u[0]/u_mag, u[1]/u_mag, u[2]/u_mag)

    # v = normal cross u
    v = (
        ny * u[2] - nz * u[1],
        nz * u[0] - nx * u[2],
        nx * u[1] - ny * u[0],
    )

    # Project vertices
    ox, oy, oz = plane_origin
    uv_pts = []
    for vx, vy, vz in vertices:
        dx, dy, dz = vx - ox, vy - oy, vz - oz
        u_coord = dx * u[0] + dy * u[1] + dz * u[2]
        v_coord = dx * v[0] + dy * v[1] + dz * v[2]
        uv_pts.append((u_coord, v_coord))

    return u, v, uv_pts


def build_slab_from_face(face_data, thickness):
    """Build a 3D slab from face data.

    The slab's outer surface matches the face, extruded inward by thickness.
    """
    normal = face_data['normal']
    vertices = face_data['vertices']
    nx, ny, nz = normal

    # We need to create a CadQuery plane at the face, draw the outline, and extrude inward.

    # Get the face's plane coordinate system
    u, v, uv_pts = project_to_plane(vertices, normal, vertices[0])

    # Create CQ plane
    origin = cq.Vector(*vertices[0])
    plane = cq.Plane(
        origin=origin,
        xDir=cq.Vector(*u),
        normal=cq.Vector(*normal),
    )

    # Draw polygon and extrude inward
    wp = cq.Workplane(plane)

    # The first point is at the plane origin (0,0 in local coords)
    # We need to use polyline with the UV coordinates
    if len(uv_pts) < 3:
        return None

    # Start from the first point (which is at 0,0)
    # polyline needs at least the points starting from current position
    wp = wp.moveTo(uv_pts[0][0], uv_pts[0][1])
    for pt in uv_pts[1:]:
        wp = wp.lineTo(pt[0], pt[1])
    wp = wp.close()

    # Extrude inward (opposite to normal)
    solid = wp.extrude(-thickness)
    return solid


def compute_bbox(solid):
    """Compute bounding box of a CadQuery solid."""
    if solid is None:
        return None
    bb = solid.val().BoundingBox()
    return {
        'xmin': bb.xmin, 'ymin': bb.ymin, 'zmin': bb.zmin,
        'xmax': bb.xmax, 'ymax': bb.ymax, 'zmax': bb.zmax,
        'dx': bb.xmax - bb.xmin, 'dy': bb.ymax - bb.ymin, 'dz': bb.zmax - bb.zmin,
    }


# ============================================================
# Test with bin_full
# ============================================================
print("=" * 70)
print("Testing approach with bin_full")
print("=" * 70)

faces, overall_bbox = get_face_data('step_files/bin_full.step')
print(f"Overall STEP bbox: X[{overall_bbox[0]:.2f},{overall_bbox[3]:.2f}] Y[{overall_bbox[1]:.2f},{overall_bbox[4]:.2f}] Z[{overall_bbox[2]:.2f},{overall_bbox[5]:.2f}]")

# Find the structural faces
bottom_outer = find_outer_face(faces, (-0.5, 0, -0.866))
right_wall_outer = find_outer_face(faces, (0.866, 0, -0.5))
front_wall = find_outer_face(faces, (0, -1, 0))
back_wall = find_outer_face(faces, (0, 1, 0))

THICKNESS_STEP = 1.2  # STEP file thickness
thickness = 3.2       # target thickness

print(f"\nBottom outer area: {bottom_outer['area']:.2f}")
print(f"Right wall outer area: {right_wall_outer['area']:.2f}")
print(f"Front wall area: {front_wall['area']:.2f}")
print(f"Back wall area: {back_wall['area']:.2f}")

# Build slabs
bottom_solid = build_slab_from_face(bottom_outer, thickness)
right_solid = build_slab_from_face(right_wall_outer, thickness)
front_solid = build_slab_from_face(front_wall, thickness)
back_solid = build_slab_from_face(back_wall, thickness)

print("\nPanel bounding boxes (thickness=3.2):")
for name, solid in [("bottom", bottom_solid), ("right_wall", right_solid),
                     ("front_wall", front_solid), ("back_wall", back_solid)]:
    bb = compute_bbox(solid)
    if bb:
        print(f"  {name}: X[{bb['xmin']:.2f},{bb['xmax']:.2f}] Y[{bb['ymin']:.2f},{bb['ymax']:.2f}] Z[{bb['zmin']:.2f},{bb['zmax']:.2f}]")
        print(f"    dims: {bb['dx']:.2f} x {bb['dy']:.2f} x {bb['dz']:.2f}")

# Now build an assembly and check the combined bounding box
assembly = cq.Assembly()
if bottom_solid:
    assembly.add(bottom_solid, name="bottom")
if right_solid:
    assembly.add(right_solid, name="right_wall")
if front_solid:
    assembly.add(front_solid, name="front_wall")
if back_solid:
    assembly.add(back_solid, name="back_wall")

# Save and check
assembly.save("output/test_approach.step")
print("\nSaved to output/test_approach.step")

# Check combined bbox
all_solids = [s for s in [bottom_solid, right_solid, front_solid, back_solid] if s is not None]
if all_solids:
    combined = all_solids[0]
    for s in all_solids[1:]:
        combined = combined.union(s)
    cbb = compute_bbox(combined)
    print(f"\nCombined bbox: X[{cbb['xmin']:.2f},{cbb['xmax']:.2f}] Y[{cbb['ymin']:.2f},{cbb['ymax']:.2f}] Z[{cbb['zmin']:.2f},{cbb['zmax']:.2f}]")
    print(f"Combined dims: {cbb['dx']:.2f} x {cbb['dy']:.2f} x {cbb['dz']:.2f}")

    # Compare with STEP
    step_dx = overall_bbox[3] - overall_bbox[0]
    step_dy = overall_bbox[4] - overall_bbox[1]
    step_dz = overall_bbox[5] - overall_bbox[2]
    print(f"\nSTEP dims:      {step_dx:.2f} x {step_dy:.2f} x {step_dz:.2f}")
    print(f"Error:          {abs(cbb['dx']-step_dx):.2f} x {abs(cbb['dy']-step_dy):.2f} x {abs(cbb['dz']-step_dz):.2f}")
