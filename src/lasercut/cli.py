"""CLI entry point for the lasercut generator."""

import argparse
import os
import shutil
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
    TAB_DIRECTION_INWARD,
    TAB_DIRECTION_OUTWARD,
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
    parser.add_argument(
        "--tab-direction",
        choices=[TAB_DIRECTION_OUTWARD, TAB_DIRECTION_INWARD],
        default=TAB_DIRECTION_INWARD,
        help=f"Positive tab style: {TAB_DIRECTION_OUTWARD} (legacy) or {TAB_DIRECTION_INWARD}",
    )
    parser.add_argument("--wall-offset", type=float, default=DEFAULT_FOLDED_OFFSET,
                        help=f"Gap from bottom plate to surrounding walls in folded layout (default: {DEFAULT_FOLDED_OFFSET}mm)")
    parser.add_argument(
        "--svg-overlay-original",
        action="store_true",
        help="Draw original projected face geometry in green on SVG for dimension debugging",
    )
    parser.add_argument(
        "--svg-verify-overlap",
        action="store_true",
        help=(
            "Write additional translucent overlap SVG panels for per-joint mesh verification "
            "(lasercut-verify-overlap.svg)"
        ),
    )
    parser.add_argument(
        "--svg-verify-overlap-baseline",
        default=None,
        help="Path to baseline overlap SVG used for regression diff/updates",
    )
    parser.add_argument(
        "--svg-verify-overlap-diff",
        action="store_true",
        help="Create heatmap-style diff images against --svg-verify-overlap-baseline",
    )
    parser.add_argument(
        "--svg-verify-overlap-update-baseline",
        action="store_true",
        help="Overwrite --svg-verify-overlap-baseline with current overlap SVG after export",
    )
    parser.add_argument(
        "--svg-verify-overlap-diff-threshold",
        type=int,
        default=10,
        help="Pixel diff threshold [0-255] for overlap heatmap (default: 10)",
    )
    parser.add_argument(
        "--svg-verify-overlap-diff-scale",
        type=float,
        default=3.0,
        help="Render scale in px/mm for overlap heatmap diff (default: 3.0)",
    )
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
    if (args.svg_verify_overlap_diff or args.svg_verify_overlap_update_baseline) and not args.svg_verify_overlap:
        print(
            "Error: --svg-verify-overlap-diff/update-baseline requires --svg-verify-overlap",
            file=sys.stderr,
        )
        sys.exit(1)
    if (args.svg_verify_overlap_diff or args.svg_verify_overlap_update_baseline) and not args.svg_verify_overlap_baseline:
        print(
            "Error: --svg-verify-overlap-baseline is required for diff/update-baseline",
            file=sys.stderr,
        )
        sys.exit(1)

    from .step_loader import load_step
    from .face_classifier import find_shared_edges, classify_faces
    from .projector import project_face
    from .finger_joints import (
        FusionJointParams,
        apply_finger_joints_fusion,
    )
    from .overlap_diff import create_overlap_diff_heatmap
    from .exporter import export_dxf, export_svg, export_svg_overlap_debug

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
    print(f"  Tab direction: {args.tab_direction}")
    if args.svg_overlay_original:
        print("  SVG original overlay: enabled")
    if args.svg_verify_overlap:
        print("  SVG overlap verifier: enabled")
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
        tab_direction=args.tab_direction,
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
                              faces=faces,
                              overlay_original=args.svg_overlay_original)
        written.append(svg_file)
        print(f"  Written: {svg_file}")

        if args.svg_verify_overlap:
            verify_path = os.path.join(args.output, "lasercut-verify-overlap.svg")
            verify_file = export_svg_overlap_debug(
                projections,
                modified_polygons,
                verify_path,
                slot_cutouts=slot_cutouts,
                shared_edges=relevant_shared,
                bottom_id=bottom.face_id,
                faces=faces,
                mesh_offset=args.thickness if args.tab_direction == TAB_DIRECTION_INWARD else 0.0,
            )
            written.append(verify_file)
            print(f"  Written: {verify_file}")

            if args.svg_verify_overlap_diff:
                baseline_path = os.path.abspath(args.svg_verify_overlap_baseline)
                if not os.path.exists(baseline_path):
                    print(
                        f"  Baseline not found (skip diff): {baseline_path}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                print("Generating overlap heatmap diff...")
                diff_artifacts = create_overlap_diff_heatmap(
                    current_svg_path=verify_file,
                    baseline_svg_path=baseline_path,
                    output_dir=args.output,
                    output_prefix="lasercut-verify-overlap",
                    px_per_mm=args.svg_verify_overlap_diff_scale,
                    diff_threshold=args.svg_verify_overlap_diff_threshold,
                )
                for key in (
                    "current_png",
                    "baseline_png",
                    "diff_gray_png",
                    "diff_mask_png",
                    "diff_heatmap_png",
                    "diff_overlay_png",
                    "diff_summary_json",
                ):
                    path = diff_artifacts[key]
                    written.append(path)
                    print(f"  Written: {path}")
                print(
                    "  Diff summary: "
                    f"{diff_artifacts['changed_pixels']} / {diff_artifacts['total_pixels']} px "
                    f"changed ({diff_artifacts['changed_ratio'] * 100:.4f}%)"
                )

            if args.svg_verify_overlap_update_baseline:
                baseline_path = os.path.abspath(args.svg_verify_overlap_baseline)
                os.makedirs(os.path.dirname(baseline_path) or ".", exist_ok=True)
                shutil.copyfile(verify_file, baseline_path)
                written.append(baseline_path)
                print(f"  Updated baseline: {baseline_path}")

    print(f"\nDone! {len(written)} file(s) written to {args.output}/")


if __name__ == "__main__":
    main()
