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
from .exporter import (
    DEFAULT_FOLDED_OFFSET,
    DEFAULT_LAYOUT,
    DEFAULT_SHEET_HEIGHT_MM,
    DEFAULT_SHEET_WIDTH_MM,
)


def _format_mm(value: float) -> str:
    """Format mm values compactly for stable filenames."""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _cut_settings_filename_suffix(thickness: float, kerf: float) -> str:
    """Build a stable cut-settings suffix for output filenames."""
    t = _format_mm(thickness)
    k = _format_mm(kerf)
    return f"_{t}mm_kerf_{k}mm"


def _settings_output_dir(root: str, thickness: float, kerf: float, layout: str) -> str:
    """Build output subfolder path grouped by thickness, kerf, and layout."""
    t = f"{_format_mm(thickness)}mm"
    k = f"kerf_{_format_mm(kerf)}mm"
    return os.path.join(root, t, k, layout)


def _verify_output_dir(root: str, model_stem: str, thickness: float, kerf: float) -> str:
    """Build verification output directory grouped by model/thickness/kerf."""
    t = f"{_format_mm(thickness)}mm"
    k = f"kerf_{_format_mm(kerf)}mm"
    return os.path.join(root, "logs", "verification", model_stem, t, k)


def _input_stem(path: str) -> str:
    """Return basename without extension for use as output filename prefix."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem or "model"


def _normalize_layout_choice(layout: str) -> str:
    if layout == "folded":
        return "unfolded"
    return layout


def _kerf_sweep(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--kerf-step must be > 0")
    if end < start:
        raise ValueError("--kerf-end must be >= --kerf-start")

    vals = []
    k = start
    # inclusive range with floating-point tolerance
    while k <= end + 1e-9:
        vals.append(round(k, 6))
        k += step

    # Preserve a clean exact end point when step lands very close.
    if abs(vals[-1] - end) > 1e-6 and vals[-1] < end:
        vals.append(round(end, 6))
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Generate laser-cut DXF/SVG files from STEP models with finger joints."
    )
    parser.add_argument("input", help="Path to STEP file")
    parser.add_argument("--thickness", type=float, default=3.0,
                        help="Material thickness in mm (default: 3.0)")
    parser.add_argument("--kerf", type=float, default=0.0,
                        help="Laser kerf in mm for compensation (default: 0.0)")
    parser.add_argument(
        "--kerfs",
        nargs="+",
        type=float,
        help="Explicit kerf sweep values in mm (overrides --kerf)",
    )
    parser.add_argument(
        "--kerf-range",
        action="store_true",
        help="Generate a kerf sweep from --kerf-start to --kerf-end",
    )
    parser.add_argument(
        "--kerf-start",
        type=float,
        default=0.0,
        help="Kerf sweep start in mm when --kerf-range is used (default: 0.0)",
    )
    parser.add_argument(
        "--kerf-end",
        type=float,
        default=0.10,
        help="Kerf sweep end in mm when --kerf-range is used (default: 0.10)",
    )
    parser.add_argument(
        "--kerf-step",
        type=float,
        default=0.02,
        help="Kerf sweep step in mm when --kerf-range is used (default: 0.02)",
    )
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
    parser.add_argument("--layout", choices=["unfolded", "packed", "both", "folded"], default=DEFAULT_LAYOUT,
                        help=f"Part layout mode (default: {DEFAULT_LAYOUT})")
    parser.add_argument("--wall-offset", type=float, default=DEFAULT_FOLDED_OFFSET,
                        help=f"Gap from bottom plate to surrounding walls in unfolded layout (default: {DEFAULT_FOLDED_OFFSET}mm)")
    parser.add_argument("--sheet-width", type=float, default=DEFAULT_SHEET_WIDTH_MM,
                        help=f"SVG sheet width in mm (default: {DEFAULT_SHEET_WIDTH_MM})")
    parser.add_argument("--sheet-height", type=float, default=DEFAULT_SHEET_HEIGHT_MM,
                        help=f"SVG sheet height in mm (default: {DEFAULT_SHEET_HEIGHT_MM})")
    parser.add_argument("--verify", action="store_true",
                        help="Run 2D seam complement verification for all joints")
    parser.add_argument("--verify-3d", action="store_true",
                        help="Run additional tab-overlap interference proxy checks")
    parser.add_argument("--verify-step", type=float, default=0.25,
                        help="Verification sample step in mm (default: 0.25)")
    parser.add_argument("--verify-tolerance", type=float, default=0.02,
                        help="Allowed seam mismatch ratio (default: 0.02)")
    parser.add_argument("--verify-interference-tolerance", type=float, default=0.01,
                        help="Allowed interference ratio for --verify-3d (default: 0.01)")
    parser.add_argument("--verify-debug", action="store_true",
                        help="Write per-joint debug SVGs for failed verification checks")
    parser.add_argument("--verify-strict", action="store_true",
                        help="Exit with non-zero status if verification fails")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    from .step_loader import load_step
    from .face_classifier import find_shared_edges, classify_faces
    from .projector import project_face
    from .finger_joints import apply_finger_joints
    from .exporter import export_dxf, export_svg
    from .verification import verify_joint_mesh, write_verification_report

    fw_display = args.finger_width if args.finger_width > 0 else DEFAULT_FINGER_WIDTH

    if args.kerfs:
        kerf_values = sorted(set(round(k, 6) for k in args.kerfs))
    elif args.kerf_range:
        kerf_values = _kerf_sweep(args.kerf_start, args.kerf_end, args.kerf_step)
    else:
        kerf_values = [round(args.kerf, 6)]

    if args.layout == "both":
        layout_modes = ["unfolded", "packed"]
    else:
        layout_modes = [_normalize_layout_choice(args.layout)]

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
    print(f"  Kerf values: {', '.join(_format_mm(k) for k in kerf_values)} mm")
    em_display = args.edge_margin if args.edge_margin >= 0 else DEFAULT_EDGE_MARGIN
    nb_display = args.notch_buffer if args.notch_buffer >= 0 else DEFAULT_NOTCH_BUFFER
    pi_display = args.plateau_inset if args.plateau_inset >= 0 else DEFAULT_PLATEAU_INSET
    mpl_display = (args.min_plateau_length if args.min_plateau_length >= 0
                   else DEFAULT_MIN_PLATEAU_LENGTH)
    print(f"  Edge margin: {em_display} mm")
    print(f"  Notch buffer: {nb_display} mm")
    print(f"  Plateau inset: {pi_display} mm")
    print(f"  Min plateau length: {mpl_display} mm")
    print(f"  Layouts: {', '.join(layout_modes)}")
    print(f"  SVG sheet: {args.sheet_width} x {args.sheet_height} mm")
    if "unfolded" in layout_modes:
        print(f"  Wall offset: {args.wall_offset} mm")
    if args.verify or args.verify_3d:
        print("  Verification: enabled")
        print(f"    Sample step: {args.verify_step} mm")
        print(f"    Mismatch tolerance: {args.verify_tolerance}")
        if args.verify_3d:
            print(f"    Interference tolerance: {args.verify_interference_tolerance}")
        print(f"    Strict mode: {'yes' if args.verify_strict else 'no'}")

    # Step 5: Export
    os.makedirs(args.output, exist_ok=True)
    written = []
    had_verify_failure = False
    file_prefix = _input_stem(args.input)

    for kerf in kerf_values:
        settings_suffix = _cut_settings_filename_suffix(args.thickness, kerf)
        print(f"\nProcessing kerf={_format_mm(kerf)} mm...")

        modified_polygons, slot_cutouts = apply_finger_joints(
            projections, relevant_shared,
            bottom_id=bottom.face_id,
            thickness=args.thickness,
            finger_width=args.finger_width,
            kerf=kerf,
            edge_margin=args.edge_margin,
            notch_buffer=args.notch_buffer,
            plateau_inset=args.plateau_inset,
            min_plateau_length=args.min_plateau_length,
            faces=faces,
            all_shared_edges=shared_edges,
        )

        # Report through-slots once per kerf setting
        for fid, slots in slot_cutouts.items():
            if slots:
                label = projections[fid].label
                print(f"  Through-slots on {label}: {len(slots)} slot(s)")

        if args.verify or args.verify_3d:
            verify_dir = _verify_output_dir(args.output, file_prefix, args.thickness, kerf)
            os.makedirs(verify_dir, exist_ok=True)
            debug_dir = os.path.join(verify_dir, "debug") if args.verify_debug else None

            report = verify_joint_mesh(
                projections=projections,
                modified_polygons=modified_polygons,
                slot_cutouts=slot_cutouts,
                shared_edges=relevant_shared,
                bottom_id=bottom.face_id,
                thickness=args.thickness,
                sample_step=args.verify_step,
                mismatch_tolerance=args.verify_tolerance,
                run_interference=args.verify_3d,
                interference_tolerance=args.verify_interference_tolerance,
                faces=faces,
                debug_dir=debug_dir,
            )

            report_path = os.path.join(
                verify_dir,
                f"{file_prefix}_{_format_mm(args.thickness)}mm_kerf_{_format_mm(kerf)}mm_verify.json",
            )
            write_verification_report(report_path, report)

            print(
                "  Verification: "
                f"{'PASS' if report.passed else 'FAIL'} "
                f"({report.total_joints - report.failed_joints}/{report.total_joints} joints)"
            )
            print(f"  Verification report: {report_path}")

            if not report.passed:
                had_verify_failure = True
                for r in report.joints:
                    if not r.passed:
                        print(
                            f"    FAIL {r.joint_type} {r.face_a_id}<->{r.face_b_id}: "
                            f"{r.reason}"
                        )
                        if r.debug_svg:
                            print(f"      debug: {r.debug_svg}")
                if args.verify_strict:
                    print("\nVerification failed and --verify-strict is set.", file=sys.stderr)
                    sys.exit(2)

        for layout_mode in layout_modes:
            output_dir = _settings_output_dir(args.output, args.thickness, kerf, layout_mode)
            os.makedirs(output_dir, exist_ok=True)

            if "dxf" in args.format:
                print(f"Exporting DXF ({layout_mode})...")
                if args.per_face:
                    dxf_files = export_dxf(projections, modified_polygons, output_dir,
                                           per_face=True, slot_cutouts=slot_cutouts,
                                           layout=layout_mode, wall_offset=args.wall_offset,
                                           shared_edges=relevant_shared,
                                           bottom_id=bottom.face_id,
                                           faces=faces,
                                           thickness=args.thickness,
                                           filename_suffix=settings_suffix)
                else:
                    dxf_path = os.path.join(output_dir, f"{file_prefix}{settings_suffix}.dxf")
                    dxf_files = export_dxf(projections, modified_polygons, dxf_path,
                                           slot_cutouts=slot_cutouts,
                                           layout=layout_mode, wall_offset=args.wall_offset,
                                           shared_edges=relevant_shared,
                                           bottom_id=bottom.face_id,
                                           faces=faces,
                                           thickness=args.thickness,
                                           filename_suffix=settings_suffix)
                written.extend(dxf_files)
                for f in dxf_files:
                    print(f"  Written: {f}")

            if "svg" in args.format:
                print(f"Exporting SVG ({layout_mode})...")
                svg_path = os.path.join(output_dir, f"{file_prefix}{settings_suffix}.svg")
                svg_file = export_svg(projections, modified_polygons, svg_path,
                                      slot_cutouts=slot_cutouts,
                                      layout=layout_mode, wall_offset=args.wall_offset,
                                      shared_edges=relevant_shared,
                                      bottom_id=bottom.face_id,
                                      faces=faces,
                                      thickness=args.thickness,
                                      filename_suffix=settings_suffix,
                                      sheet_width=args.sheet_width,
                                      sheet_height=args.sheet_height)
                written.append(svg_file)
                print(f"  Written: {svg_file}")

    print(
        f"\nDone! {len(written)} file(s) written "
        f"across {len(kerf_values)} kerf setting(s) and {len(layout_modes)} layout(s)."
    )
    if had_verify_failure and not args.verify_strict:
        print("Warning: Verification reported failed joints. See reports in output/logs/verification.")


if __name__ == "__main__":
    main()
