"""CLI entry point for the lasercut generator."""

import argparse
import os
import sys

from .finger_joints import (
    DEFAULT_MIN_PLATEAU_LENGTH,
    DEFAULT_NOTCH_BUFFER,
    DEFAULT_PLATEAU_INSET,
    FUSION_DYNAMIC_EQUAL,
    FUSION_DYNAMIC_FIXED_FINGER,
    FUSION_DYNAMIC_FIXED_NOTCH,
    FUSION_DEFAULT_EDGE_MARGIN,
    FUSION_PLACEMENT_FINGERS_OUTSIDE,
    FUSION_PLACEMENT_NOTCHES_OUTSIDE,
    FUSION_PLACEMENT_SAME_START_FINGER,
    FUSION_PLACEMENT_SAME_START_NOTCH,
)
from .exporter import DEFAULT_FOLDED_OFFSET, DEFAULT_LAYOUT


def main():
    parser = argparse.ArgumentParser(
        description="Generate laser-cut DXF/SVG files from STEP models with finger joints."
    )
    parser.add_argument("input", help="Path to STEP file")
    parser.add_argument("--thickness", type=float, default=3.0,
                        help="Material thickness in mm (default: 3.0)")
    parser.add_argument("--kerf", type=float, default=0.0,
                        help="Laser kerf in mm for compensation (default: 0.0)")
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
                        help=f"Safe zone at edge ends in mm, -1 = auto ({FUSION_DEFAULT_EDGE_MARGIN}mm)")
    parser.add_argument("--notch-buffer", type=float, default=-1,
                        help=f"Safe zone around notches in mm, -1 = auto ({DEFAULT_NOTCH_BUFFER}mm)")
    parser.add_argument("--plateau-inset", type=float, default=-1,
                        help=f"Inset from plateau boundaries in mm, -1 = auto ({DEFAULT_PLATEAU_INSET}mm)")
    parser.add_argument("--min-plateau-length", type=float, default=-1,
                        help=f"Minimum plateau segment length in mm, -1 = auto ({DEFAULT_MIN_PLATEAU_LENGTH}mm)")
    parser.add_argument("--layout", choices=["folded", "packed"], default=DEFAULT_LAYOUT,
                        help=f"Part layout mode (default: {DEFAULT_LAYOUT})")
    parser.add_argument("--wall-offset", type=float, default=DEFAULT_FOLDED_OFFSET,
                        help=f"Gap from bottom plate to surrounding walls in folded layout (default: {DEFAULT_FOLDED_OFFSET}mm)")
    parser.add_argument(
        "--fusion-placement",
        choices=[
            FUSION_PLACEMENT_FINGERS_OUTSIDE,
            FUSION_PLACEMENT_NOTCHES_OUTSIDE,
            FUSION_PLACEMENT_SAME_START_FINGER,
            FUSION_PLACEMENT_SAME_START_NOTCH,
        ],
        default=FUSION_PLACEMENT_FINGERS_OUTSIDE,
        help="Finger placement mode (default: fingers_outside)",
    )
    parser.add_argument(
        "--fusion-size-mode",
        choices=[FUSION_DYNAMIC_EQUAL, FUSION_DYNAMIC_FIXED_NOTCH, FUSION_DYNAMIC_FIXED_FINGER],
        default=FUSION_DYNAMIC_EQUAL,
        help="Finger sizing mode (default: equal)",
    )
    parser.add_argument(
        "--fusion-count-mode",
        choices=["dynamic", "fixed"],
        default="dynamic",
        help="Finger count mode (default: dynamic)",
    )
    parser.add_argument(
        "--fusion-fixed-num-fingers",
        type=int,
        default=3,
        help="Fixed number of fingers (used when --fusion-count-mode fixed)",
    )
    parser.add_argument(
        "--fusion-fixed-finger-size",
        type=float,
        default=20.0,
        help="Fixed finger size in mm (used by fixed_finger size mode)",
    )
    parser.add_argument(
        "--fusion-fixed-notch-size",
        type=float,
        default=20.0,
        help="Fixed notch size in mm (used by fixed_notch size mode)",
    )
    parser.add_argument(
        "--fusion-min-finger-size",
        type=float,
        default=20.0,
        help="Minimum finger size in mm for dynamic sizing",
    )
    parser.add_argument(
        "--fusion-min-notch-size",
        type=float,
        default=20.0,
        help="Minimum notch size in mm for dynamic sizing",
    )
    parser.add_argument(
        "--fusion-gap",
        type=float,
        default=0.0,
        help="Gap between fingers/notches in mm",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    from .step_loader import load_step
    from .face_classifier import find_shared_edges, classify_faces
    from .projector import project_face
    from .finger_joints import (
        FusionJointParams,
        apply_finger_joints_fusion,
    )
    from .exporter import export_dxf, export_svg

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

    em_display = args.edge_margin if args.edge_margin >= 0 else FUSION_DEFAULT_EDGE_MARGIN
    nb_display = args.notch_buffer if args.notch_buffer >= 0 else DEFAULT_NOTCH_BUFFER
    pi_display = args.plateau_inset if args.plateau_inset >= 0 else DEFAULT_PLATEAU_INSET
    mpl_display = (args.min_plateau_length if args.min_plateau_length >= 0
                   else DEFAULT_MIN_PLATEAU_LENGTH)

    print(f"Applying finger joints ({len(relevant_shared)} shared edges)...")
    print(f"  Thickness: {args.thickness} mm")
    print(f"  Kerf: {args.kerf} mm")
    print(f"  Edge margin: {em_display} mm")
    print(f"  Notch buffer: {nb_display} mm")
    print(f"  Plateau inset: {pi_display} mm")
    print(f"  Min plateau length: {mpl_display} mm")
    print(f"  Layout: {args.layout}")
    print(f"  Placement: {args.fusion_placement}")
    print(f"  Size mode: {args.fusion_size_mode}")
    print(f"  Count mode: {args.fusion_count_mode}")
    if args.fusion_count_mode == "fixed":
        print(f"  Fixed finger count: {args.fusion_fixed_num_fingers}")
    print(f"  Min finger size: {args.fusion_min_finger_size} mm")
    print(f"  Min notch size: {args.fusion_min_notch_size} mm")
    print(f"  Gap: {args.fusion_gap} mm")
    if args.layout == "folded":
        print(f"  Wall offset: {args.wall_offset} mm")

    fusion_params = FusionJointParams(
        placement_type=args.fusion_placement,
        dynamic_size_type=args.fusion_size_mode,
        is_number_of_fingers_fixed=(args.fusion_count_mode == "fixed"),
        fixed_num_fingers=args.fusion_fixed_num_fingers,
        fixed_finger_size=args.fusion_fixed_finger_size,
        fixed_notch_size=args.fusion_fixed_notch_size,
        min_finger_size=args.fusion_min_finger_size,
        min_notch_size=args.fusion_min_notch_size,
        gap=args.fusion_gap,
    )
    modified_polygons, slot_cutouts = apply_finger_joints_fusion(
        projections,
        relevant_shared,
        bottom_id=bottom.face_id,
        thickness=args.thickness,
        kerf=args.kerf,
        edge_margin=args.edge_margin,
        notch_buffer=args.notch_buffer,
        plateau_inset=args.plateau_inset,
        min_plateau_length=args.min_plateau_length,
        faces=faces,
        fusion_params=fusion_params,
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
                                   per_face=True, slot_cutouts=slot_cutouts,
                                   layout=args.layout, wall_offset=args.wall_offset,
                                   shared_edges=relevant_shared,
                                   bottom_id=bottom.face_id,
                                   faces=faces)
        else:
            dxf_path = os.path.join(args.output, "lasercut.dxf")
            dxf_files = export_dxf(projections, modified_polygons, dxf_path,
                                   slot_cutouts=slot_cutouts,
                                   layout=args.layout, wall_offset=args.wall_offset,
                                   shared_edges=relevant_shared,
                                   bottom_id=bottom.face_id,
                                   faces=faces)
        written.extend(dxf_files)
        for f in dxf_files:
            print(f"  Written: {f}")

    if "svg" in args.format:
        print("Exporting SVG...")
        svg_path = os.path.join(args.output, "lasercut.svg")
        svg_file = export_svg(projections, modified_polygons, svg_path,
                              slot_cutouts=slot_cutouts,
                              layout=args.layout, wall_offset=args.wall_offset,
                              shared_edges=relevant_shared,
                              bottom_id=bottom.face_id,
                              faces=faces)
        written.append(svg_file)
        print(f"  Written: {svg_file}")

    print(f"\nDone! {len(written)} file(s) written to {args.output}/")


if __name__ == "__main__":
    main()
