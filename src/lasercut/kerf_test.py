"""Generate kerf calibration coupons as an SVG sheet."""

import argparse
import math
import os

import svgwrite
from shapely.geometry import MultiPolygon, box


def _format_mm(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _add_geom(dwg, group, geom, offset_x: float, offset_y: float, stroke_width: float):
    if geom.is_empty:
        return

    geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for poly in geoms:
        outer = [(x + offset_x, y + offset_y) for x, y in poly.exterior.coords[:-1]]
        group.add(
            dwg.polygon(
                outer,
                fill="none",
                stroke="black",
                stroke_width=stroke_width,
            )
        )
        for ring in poly.interiors:
            hole = [(x + offset_x, y + offset_y) for x, y in ring.coords[:-1]]
            group.add(
                dwg.polygon(
                    hole,
                    fill="none",
                    stroke="black",
                    stroke_width=stroke_width,
                )
            )


def _tab_coupon(
    joint_width: float,
    base_w: float,
    base_h: float,
    tab_d: float,
    segment_count: int = 5,
    margin: float = 1.2,
):
    """Single-edge multi-finger tab coupon for fit testing."""
    _ = joint_width  # kept for CLI/API compatibility and future scaling

    shape = box(0.0, 0.0, base_w, base_h)
    usable = base_h - 2.0 * margin
    seg = usable / segment_count

    for i in range(0, segment_count, 2):
        y0 = margin + i * seg
        y1 = margin + (i + 1) * seg
        shape = shape.union(box(base_w, y0, base_w + tab_d, y1))

    return shape


def _slot_coupon(
    joint_width: float,
    base_w: float,
    base_h: float,
    slot_d: float,
    segment_count: int = 5,
    margin: float = 1.2,
):
    """Single-edge multi-finger slot coupon mating the tab coupon."""
    _ = joint_width  # kept for CLI/API compatibility and future scaling

    shape = box(0.0, 0.0, base_w, base_h)
    usable = base_h - 2.0 * margin
    seg = usable / segment_count

    for i in range(0, segment_count, 2):
        y0 = margin + i * seg
        y1 = margin + (i + 1) * seg
        shape = shape.difference(box(0.0, y0, slot_d, y1))

    return shape


def _apply_kerf(shape, kerf: float):
    if abs(kerf) < 1e-9:
        return shape
    compensated = shape.buffer(kerf / 2.0, join_style="mitre")
    if compensated.is_empty:
        return shape
    if isinstance(compensated, MultiPolygon):
        compensated = max(compensated.geoms, key=lambda g: g.area)
    return compensated


def generate_kerf_test_svg(
    output_path: str,
    joint_width: float,
    kerfs: list[float],
    columns: int = 3,
    stroke_width: float = 0.5,
) -> str:
    """Generate a compact kerf test sheet with tab/slot coupon pairs."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not kerfs:
        raise ValueError("Need at least one kerf value")
    if columns < 1:
        raise ValueError("columns must be >= 1")

    # Compact grid layout: each cell contains exactly two mating pieces.
    pad = 3.0
    col_gap = 2.0
    row_gap = 2.0
    piece_gap = 2.0
    piece_w = 26.0
    piece_h = 26.0
    finger_d = 10.0  # tab protrusion / slot depth

    right_piece_dx = piece_w + piece_gap + finger_d
    cell_w = right_piece_dx + piece_w
    cell_h = piece_h

    cols = min(columns, len(kerfs))
    rows = math.ceil(len(kerfs) / cols)
    width = pad * 2 + cols * cell_w + max(0, cols - 1) * col_gap
    height = pad * 2 + rows * cell_h + max(0, rows - 1) * row_gap

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{width:.1f}mm", f"{height:.1f}mm"),
        viewBox=f"0 0 {width:.1f} {height:.1f}",
    )
    dwg.set_desc(
        title="Kerf Test Coupons",
        desc=(
            "Thickness-agnostic compact sheet. Each cell has two mating "
            "multi-finger joint sides (3 fingers): tab piece (left) and "
            "slot piece (right), labeled by kerf. Includes looser variants."
        ),
    )

    for idx, kerf in enumerate(kerfs):
        row = idx // cols
        col = idx % cols
        origin_x = pad + col * (cell_w + col_gap)
        origin_y = pad + row * (cell_h + row_gap)

        group = dwg.g(id=f"k{_format_mm(kerf)}")

        tab = _apply_kerf(
            _tab_coupon(joint_width, base_w=piece_w, base_h=piece_h, tab_d=finger_d),
            kerf,
        )
        slot = _apply_kerf(
            _slot_coupon(joint_width, base_w=piece_w, base_h=piece_h, slot_d=finger_d),
            kerf,
        )

        _add_geom(dwg, group, tab, origin_x, origin_y, stroke_width)
        _add_geom(dwg, group, slot, origin_x + right_piece_dx, origin_y, stroke_width)

        # Keep kerf label on both pieces; place away from joint edges so
        # labels stay clear of finger/slot features.
        kerf_label = f"k{_format_mm(kerf)}"
        label_positions = (
            origin_x + (piece_w * 0.34),                   # left piece: away from right joint edge
            origin_x + right_piece_dx + (piece_w * 0.72),  # right piece: away from left slot edge
        )
        for cx in label_positions:
            cy = origin_y + (piece_h * 0.52)
            group.add(
                dwg.text(
                    kerf_label,
                    insert=(cx, cy),
                    text_anchor="middle",
                    dominant_baseline="middle",
                    font_size="7.0",
                    font_family="monospace",
                    fill="blue",
                    transform=f"rotate(-90 {cx} {cy})",
                )
            )
        dwg.add(group)

    dwg.save()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate kerf test coupon SVG sheet.")
    parser.add_argument(
        "--joint-width",
        type=float,
        default=3.2,
        help="Nominal joint width in mm for coupon geometry (default: 3.2)",
    )
    parser.add_argument(
        "--kerfs",
        nargs="+",
        type=float,
        default=[0.04, 0.06, 0.08, 0.10, 0.12, 0.14],
        help="Kerf values in mm (default: 0.04 0.06 0.08 0.10 0.12 0.14)",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of kerf cells per row (default: 3)",
    )
    parser.add_argument(
        "--output",
        default="output/kerf_test_compact.svg",
        help="Output SVG path (default: output/kerf_test_compact.svg)",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=0.5,
        help="Cut line stroke width in SVG units/mm (default: 0.5)",
    )

    args = parser.parse_args()
    out = generate_kerf_test_svg(
        output_path=args.output,
        joint_width=args.joint_width,
        kerfs=args.kerfs,
        columns=args.columns,
        stroke_width=args.stroke_width,
    )
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
