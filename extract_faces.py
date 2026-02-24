"""Extract exact face geometry from STEP reference files."""

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
import json


def extract_faces(step_path):
    """Extract all planar faces from a STEP file."""
    shape = cq.importers.importStep(str(step_path))
    solid = shape.val()
    occ_shape = solid.wrapped

    # Overall bounding box
    bbox = Bnd_Box()
    BRepBndLib.Add_s(occ_shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    print(f"\nOverall bounding box:")
    print(f"  X: [{xmin:.2f}, {xmax:.2f}] ({xmax-xmin:.2f}mm)")
    print(f"  Y: [{ymin:.2f}, {ymax:.2f}] ({ymax-ymin:.2f}mm)")
    print(f"  Z: [{zmin:.2f}, {zmax:.2f}] ({zmax-zmin:.2f}mm)")

    faces = []
    exp = TopExp_Explorer(occ_shape, TopAbs_FACE)
    face_id = 0

    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        surf = BRepAdaptor_Surface(face)

        if surf.GetType() == GeomAbs_Plane:
            plane = surf.Plane()
            normal_dir = plane.Axis().Direction()
            loc = plane.Location()

            if face.Orientation() == TopAbs_REVERSED:
                n = (-normal_dir.X(), -normal_dir.Y(), -normal_dir.Z())
            else:
                n = (normal_dir.X(), normal_dir.Y(), normal_dir.Z())

            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            area = props.Mass()
            cp = props.CentreOfMass()

            # Face bounding box
            fbbox = Bnd_Box()
            BRepBndLib.Add_s(face, fbbox)
            fxmin, fymin, fzmin, fxmax, fymax, fzmax = fbbox.Get()

            # Get outer wire vertices
            outer_wire = BRepTools.OuterWire_s(face)
            wire_exp = BRepTools_WireExplorer(outer_wire, face)
            vertices = []
            while wire_exp.More():
                vtx = wire_exp.CurrentVertex()
                pt = BRep_Tool.Pnt_s(vtx)
                vertices.append((round(pt.X(), 2), round(pt.Y(), 2), round(pt.Z(), 2)))
                wire_exp.Next()

            # Get the plane's X and Y directions
            ax = plane.Position()
            x_dir = ax.XDirection()
            y_dir = ax.YDirection()

            faces.append({
                'id': face_id,
                'normal': (round(n[0], 4), round(n[1], 4), round(n[2], 4)),
                'area': round(area, 2),
                'center': (round(cp.X(), 2), round(cp.Y(), 2), round(cp.Z(), 2)),
                'plane_origin': (round(loc.X(), 2), round(loc.Y(), 2), round(loc.Z(), 2)),
                'x_dir': (round(x_dir.X(), 6), round(x_dir.Y(), 6), round(x_dir.Z(), 6)),
                'y_dir': (round(y_dir.X(), 6), round(y_dir.Y(), 6), round(y_dir.Z(), 6)),
                'bbox': {
                    'xmin': round(fxmin, 2), 'ymin': round(fymin, 2), 'zmin': round(fzmin, 2),
                    'xmax': round(fxmax, 2), 'ymax': round(fymax, 2), 'zmax': round(fzmax, 2),
                },
                'n_vertices': len(vertices),
                'vertices': vertices,
            })
            face_id += 1

        exp.Next()

    return faces


def classify_face(face):
    """Classify a face by its normal direction."""
    nx, ny, nz = face['normal']
    anx, any_, anz = abs(nx), abs(ny), abs(nz)

    # Group by normal direction
    if any_ > 0.9:
        return f"end_wall_{'back' if ny > 0 else 'front'}"
    elif abs(nx - 0.5) < 0.1 and abs(nz - 0.866) < 0.1:
        return "bottom_inner"
    elif abs(nx + 0.5) < 0.1 and abs(nz + 0.866) < 0.1:
        return "bottom_outer"
    elif abs(nx - 0.866) < 0.1 and abs(nz + 0.5) < 0.1:
        return "right_wall_outer"
    elif abs(nx + 0.866) < 0.1 and abs(nz - 0.5) < 0.1:
        return "right_wall_inner"
    elif abs(nx + 0.866) < 0.1 and abs(nz + 0.5) < 0.1:
        return "left_wall_outer"
    elif abs(nx - 0.866) < 0.1 and abs(nz - 0.5) < 0.1:
        return "left_wall_inner"
    else:
        return f"other_n=({nx:.3f},{ny:.3f},{nz:.3f})"


def analyze_step(step_path):
    """Full analysis of a STEP file."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {step_path}")
    print(f"{'='*70}")

    faces = extract_faces(step_path)

    # Sort by area descending
    faces.sort(key=lambda f: f['area'], reverse=True)

    print(f"\nFound {len(faces)} planar faces")
    print(f"\nTop faces by area:")

    for f in faces[:20]:
        cls = classify_face(f)
        print(f"\n  Face {f['id']}: {cls}")
        print(f"    Normal: {f['normal']}")
        print(f"    Area: {f['area']:.2f}")
        print(f"    Center: {f['center']}")
        print(f"    Plane origin: {f['plane_origin']}")
        print(f"    X_dir: {f['x_dir']}")
        print(f"    Y_dir: {f['y_dir']}")
        bb = f['bbox']
        print(f"    BBox: X[{bb['xmin']:.2f},{bb['xmax']:.2f}] Y[{bb['ymin']:.2f},{bb['ymax']:.2f}] Z[{bb['zmin']:.2f},{bb['zmax']:.2f}]")
        print(f"    Vertices ({f['n_vertices']}):")
        for v in f['vertices']:
            print(f"      {v}")

    return faces


# Analyze bin_full
faces_full = analyze_step('step_files/bin_full.step')

# Save JSON for further analysis
data = {
    'bin_full': faces_full,
}

print("\n\n--- Analyzing bin_third_left ---")
faces_tl = analyze_step('step_files/bin_third_left.step')
data['bin_third_left'] = faces_tl

print("\n\n--- Analyzing bin_third_center ---")
faces_tc = analyze_step('step_files/bin_third_center.step')
data['bin_third_center'] = faces_tc

with open('output/face_analysis.json', 'w') as f:
    json.dump(data, f, indent=2)
print("\nSaved to output/face_analysis.json")
