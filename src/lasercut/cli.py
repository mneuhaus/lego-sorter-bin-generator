"""CLI entry point for lasercut generator."""

import argparse
import os

from lasercut.panels import load_step_panels
from lasercut.joints import apply_finger_joints
from lasercut.exporter import export_svg


def _num_token(value: float) -> str:
    """Human-readable numeric token for filenames (e.g. 3.2 -> 3.2)."""
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return s


def main():
    parser = argparse.ArgumentParser(description="Generate lasercut SVG from STEP file")
    parser.add_argument("step_file", help="Path to STEP file")
    parser.add_argument("--thickness", type=float, default=3.2, help="Material thickness in mm")
    parser.add_argument("--finger-width", type=float, default=20.0, help="Target finger width in mm")
    parser.add_argument(
        "--living-hinge-angle",
        type=float,
        default=45.0,
        help=(
            "Use living-hinge slits instead of fingers on non-bottom seams below this "
            "angle in degrees (<= 0 disables)"
        ),
    )
    parser.add_argument(
        "--kerf",
        type=float,
        default=0.0,
        help="Kerf compensation in mm (positive=tighter fit, negative=looser)",
    )
    parser.add_argument(
        "--layout",
        choices=["unfolded", "packed"],
        default="unfolded",
        help="Layout mode for SVG output",
    )
    parser.add_argument("--sheet-width", type=float, help="Sheet width in mm (packed layout)")
    parser.add_argument("--sheet-height", type=float, help="Sheet height in mm (packed layout)")
    parser.add_argument("--part-gap", type=float, default=4.0, help="Gap between packed parts in mm")
    parser.add_argument("--sheet-gap", type=float, default=15.0, help="Gap between sheets in SVG in mm")
    parser.add_argument(
        "--pack-rotations",
        type=int,
        default=2,
        help="Rotation steps for packed layout (1=no rotation, 2=0/90, N=360/N steps)",
    )
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    if args.layout == "packed":
        if args.sheet_width is None or args.sheet_height is None:
            parser.error("--sheet-width and --sheet-height are required for --layout packed")
        if args.sheet_width <= 0 or args.sheet_height <= 0:
            parser.error("--sheet-width and --sheet-height must be > 0")

    original_model = load_step_panels(args.step_file, args.thickness)
    model = apply_finger_joints(
        original_model,
        args.finger_width,
        kerf=args.kerf,
        living_hinge_angle_threshold_deg=args.living_hinge_angle,
    )

    thickness_label = f"{_num_token(args.thickness)}mm"
    kerf_label = f"k{_num_token(args.kerf)}mm"
    if args.layout == "packed":
        sw = int(round(args.sheet_width)) if args.sheet_width is not None else 0
        sh = int(round(args.sheet_height)) if args.sheet_height is not None else 0
        folder_name = f"bins_{thickness_label}_{kerf_label}_{sw}x{sh}"
    else:
        folder_name = f"bins_{thickness_label}_{kerf_label}"

    output_dir = os.path.join(args.output, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.step_file))[0]
    filename_parts = [name, args.layout, thickness_label, kerf_label]
    if args.layout == "packed":
        filename_parts.append(f"{sw}x{sh}")
    output_file = "-".join(filename_parts) + ".svg"
    output_path = os.path.join(output_dir, output_file)
    export_svg(
        model,
        output_path,
        reference_model=original_model,
        layout=args.layout,
        sheet_width=args.sheet_width,
        sheet_height=args.sheet_height,
        part_gap=args.part_gap,
        sheet_gap=args.sheet_gap,
        pack_rotations=args.pack_rotations,
    )
    print(f"SVG written to {output_path}")


if __name__ == "__main__":
    main()
