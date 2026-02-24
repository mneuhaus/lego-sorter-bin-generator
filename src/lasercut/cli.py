"""CLI entry point for lasercut generator."""

import argparse
import os

from lasercut.panels import load_step_panels
from lasercut.joints import apply_finger_joints
from lasercut.exporter import export_svg


def main():
    parser = argparse.ArgumentParser(description="Generate lasercut SVG from STEP file")
    parser.add_argument("step_file", help="Path to STEP file")
    parser.add_argument("--thickness", type=float, default=3.2, help="Material thickness in mm")
    parser.add_argument("--finger-width", type=float, default=20.0, help="Target finger width in mm")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    original_model = load_step_panels(args.step_file, args.thickness)
    model = apply_finger_joints(original_model, args.finger_width)

    os.makedirs(args.output, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.step_file))[0]
    output_path = os.path.join(args.output, f"{name}.svg")
    export_svg(model, output_path, reference_model=original_model)
    print(f"SVG written to {output_path}")


if __name__ == "__main__":
    main()
