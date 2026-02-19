"""CLI entry point for the lasercut generator."""

import argparse
import os
import sys

from .finger_joints import (
    DEFAULT_EDGE_MARGIN,
    DEFAULT_FINGER_WIDTH,
    DEFAULT_MIN_PLATEAU_LENGTH,
    DEFAULT_NOTCH_BUFFER,
    DEFAULT_PLATEAU_INSET,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate laser-cut DXF/SVG files from STEP models with finger joints."
    )
    parser.add_argument("input", help="Path to STEP file")
    parser.add_argument("--thickness", type=float, default=3.0,
                        help="Material thickness in mm (default: 3.0)")
    parser.add_argument("--kerf", type=float, default=0.0,
                        help="Laser kerf in mm for compensation (default: 0.0)")
    parser.add_argument("--finger-width", type=float, default=0,
                        help=f"Finger width in mm, 0 = auto ({DEFAULT_FINGER_WIDTH}mm)")
    parser.add_argument("--format", nargs="+", default=["dxf", "svg"],
                        choices=["dxf", "svg"],
                        help="Output formats (default: dxf svg)")
    parser.add_argument("--output", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--min-area", type=float, default=100.0,
                        help="Minimum face area in mm² to include (default: 100)")
    parser.add_argument("--per-face", action="store_true",
                        help="Write one DXF file per face")
    parser.add_argument("--body", type=int, default=0,
                        help="Which body to process (0-based index, -1 for all, default: 0)")
    parser.add_argument("--edge-margin", type=float, default=-1,
                        help=f"Safe zone at edge ends in mm, -1 = auto ({DEFAULT_EDGE_MARGIN}mm)")
    parser.add_argument("--notch-buffer", type=float, default=-1,
                        help=f"Safe zone around notches in mm, -1 = auto ({DEFAULT_NOTCH_BUFFER}mm)")
    parser.add_argument("--plateau-inset", type=float, default=-1,
                        help=f"Inset from plateau boundaries in mm, -1 = auto ({DEFAULT_PLATEAU_INSET}mm)")
    parser.add_argument("--min-plateau-length", type=float, default=-1,
                        help=f"Minimum plateau segment length in mm, -1 = auto ({DEFAULT_MIN_PLATEAU_LENGTH}mm)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    from .step_loader import load_step
    from .face_classifier import find_shared_edges, classify_faces
    from .projector import project_face
    from .finger_joints import apply_finger_joints
    from .exporter import export_dxf, export_svg

    fw_display = args.finger_width if args.finger_width > 0 else DEFAULT_FINGER_WIDTH

    # Step 1: Load STEP and extract planar faces
    print(f"Loading STEP file: {args.input}")
    faces = load_step(args.input, min_area=args.min_area, body_index=args.body)
    print(f"  Found {len(faces)} planar faces above {args.min_area} mm² threshold")

    if not faces:
        print("No qualifying faces found. Try lowering --min-area.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Find shared edges and classify faces
    print("Finding shared edges...")
    shared_edges = find_shared_edges(faces)
    print(f"  Found {len(shared_edges)} shared edges")

    print("Classifying faces...")
    classification = classify_faces(faces, shared_edges)
    bottom = classification['bottom']
    walls = classification['walls']

    print(f"  Bottom plate: face {bottom.face_id} (area={bottom.area:.1f} mm²)")
    print(f"  Walls: {len(walls)} faces")
    for w in walls:
        print(f"    Face {w.face_id}: area={w.area:.1f} mm², "
              f"normal=({w.normal[0]:.2f}, {w.normal[1]:.2f}, {w.normal[2]:.2f})")

    if classification['other']:
        print(f"  Other faces (not connected to bottom): {len(classification['other'])}")

    # Step 3: Project faces to 2D
    print("Projecting faces to 2D...")
    relevant_faces = [bottom] + walls
    projections = {}

    for face in relevant_faces:
        if face.face_id == bottom.face_id:
            label = "bottom"
        else:
            nx, ny, nz = face.normal
            if abs(nx) > abs(ny) and abs(nx) > abs(nz):
                label = f"wall_x_{'pos' if nx > 0 else 'neg'}_{face.face_id}"
            elif abs(ny) > abs(nz):
                label = f"wall_y_{'pos' if ny > 0 else 'neg'}_{face.face_id}"
            else:
                label = f"wall_z_{'pos' if nz > 0 else 'neg'}_{face.face_id}"

        proj = project_face(face, label=label)
        projections[face.face_id] = proj
        print(f"  {label}: {len(proj.outer_polygon)} vertices, "
              f"{len(proj.inner_polygons)} holes")

    # Step 4: Apply finger joints (edge joints + through-slots)
    relevant_ids = set(projections.keys())
    relevant_shared = [se for se in shared_edges
                       if se.face_a_id in relevant_ids and se.face_b_id in relevant_ids]

    print(f"Applying finger joints ({len(relevant_shared)} shared edges)...")
    print(f"  Thickness: {args.thickness} mm")
    print(f"  Finger width: {fw_display} mm")
    print(f"  Kerf: {args.kerf} mm")
    em_display = args.edge_margin if args.edge_margin >= 0 else DEFAULT_EDGE_MARGIN
    nb_display = args.notch_buffer if args.notch_buffer >= 0 else DEFAULT_NOTCH_BUFFER
    pi_display = args.plateau_inset if args.plateau_inset >= 0 else DEFAULT_PLATEAU_INSET
    mpl_display = (args.min_plateau_length if args.min_plateau_length >= 0
                   else DEFAULT_MIN_PLATEAU_LENGTH)
    print(f"  Edge margin: {em_display} mm")
    print(f"  Notch buffer: {nb_display} mm")
    print(f"  Plateau inset: {pi_display} mm")
    print(f"  Min plateau length: {mpl_display} mm")

    modified_polygons, slot_cutouts = apply_finger_joints(
        projections, relevant_shared,
        bottom_id=bottom.face_id,
        thickness=args.thickness,
        finger_width=args.finger_width,
        kerf=args.kerf,
        edge_margin=args.edge_margin,
        notch_buffer=args.notch_buffer,
        plateau_inset=args.plateau_inset,
        min_plateau_length=args.min_plateau_length,
        faces=faces,
        all_shared_edges=shared_edges,
    )

    # Report through-slots
    for fid, slots in slot_cutouts.items():
        if slots:
            label = projections[fid].label
            print(f"  Through-slots on {label}: {len(slots)} slot(s)")

    # Step 5: Export
    os.makedirs(args.output, exist_ok=True)
    written = []

    if "dxf" in args.format:
        print("Exporting DXF...")
        if args.per_face:
            dxf_files = export_dxf(projections, modified_polygons, args.output,
                                   per_face=True, slot_cutouts=slot_cutouts)
        else:
            dxf_path = os.path.join(args.output, "lasercut.dxf")
            dxf_files = export_dxf(projections, modified_polygons, dxf_path,
                                   slot_cutouts=slot_cutouts)
        written.extend(dxf_files)
        for f in dxf_files:
            print(f"  Written: {f}")

    if "svg" in args.format:
        print("Exporting SVG...")
        svg_path = os.path.join(args.output, "lasercut.svg")
        svg_file = export_svg(projections, modified_polygons, svg_path,
                              slot_cutouts=slot_cutouts)
        written.append(svg_file)
        print(f"  Written: {svg_file}")

    print(f"\nDone! {len(written)} file(s) written to {args.output}/")


if __name__ == "__main__":
    main()
