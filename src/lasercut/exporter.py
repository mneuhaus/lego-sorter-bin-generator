"""DXF and SVG export for laser cutting."""

import ezdxf
import svgwrite

from .projector import Projection2D

DEFAULT_LAYOUT = "folded"
DEFAULT_FOLDED_OFFSET = 10.0


def _pack_parts(parts: list[dict], padding: float = 5) -> tuple[list[tuple], float, float]:
    """Pack parts efficiently using a bottom-left placement algorithm.

    Tries each part in both orientations (original and 90-degree rotated)
    and places it in the position that wastes the least vertical space.

    Returns:
        (positions, total_width, total_height) where positions is
        [(x_offset, y_offset, rotated), ...]
    """
    indexed_parts = list(enumerate(parts))
    indexed_parts.sort(key=lambda ip: max(ip[1]["width"], ip[1]["height"]), reverse=True)

    placed = []  # (x, y, w, h, original_index, rotated)

    for orig_idx, part in indexed_parts:
        w0, h0 = part["width"], part["height"]

        best_pos = None
        best_y_max = float("inf")
        best_rotated = False

        for rotated, (w, h) in [(False, (w0, h0)), (True, (h0, w0))]:
            x_candidates = [0] + [p[0] + p[2] + padding for p in placed]

            for x in x_candidates:
                y = 0
                for px, py, pw, ph, _, _ in placed:
                    if x < px + pw + padding and x + w + padding > px:
                        y = max(y, py + ph + padding)

                y_max = y + h
                if y_max < best_y_max:
                    best_y_max = y_max
                    best_pos = (x, y, w, h)
                    best_rotated = rotated

        if best_pos:
            x, y, w, h = best_pos
            placed.append((x, y, w, h, orig_idx, best_rotated))

    positions = [None] * len(parts)
    for x, y, w, h, orig_idx, rotated in placed:
        positions[orig_idx] = (x + padding, y + padding, rotated)

    total_w = max((x + w for x, y, w, h, _, _ in placed), default=0) + 2 * padding
    total_h = max((y + h for x, y, w, h, _, _ in placed), default=0) + 2 * padding

    return positions, total_w, total_h


def _rotate_polygon_90(polygon, w, h):
    """Rotate a polygon 90 degrees clockwise. (x,y) -> (y, w-x)."""
    return [(y, w - x) for x, y in polygon]


def _prepare_parts(
    projections: dict[int, Projection2D],
    modified_polygons: dict[int, list[tuple[float, float]]],
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] | None,
) -> list[dict]:
    """Normalize all part geometry to local (0,0)-based coordinates."""
    if slot_cutouts is None:
        slot_cutouts = {}

    parts = []
    for fid in sorted(modified_polygons.keys()):
        polygon = modified_polygons[fid]
        proj = projections[fid]
        if not polygon:
            continue

        all_polys = [polygon] + slot_cutouts.get(fid, [])
        xs = []
        ys = []
        for poly in all_polys:
            xs.extend(p[0] for p in poly)
            ys.extend(p[1] for p in poly)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        norm_poly = [(x - min_x, y - min_y) for x, y in polygon]
        norm_holes = [[(x - min_x, y - min_y) for x, y in hole] for hole in slot_cutouts.get(fid, [])]

        parts.append(
            {
                "fid": fid,
                "label": proj.label or f"face_{fid}",
                "polygon": norm_poly,
                "holes": norm_holes,
                "width": max_x - min_x,
                "height": max_y - min_y,
            }
        )

    return parts


def _part_side(label: str) -> str:
    """Map a projected face label to a folded-layout side bucket."""
    if label == "bottom":
        return "center"
    if label.startswith("wall_y_pos_") or label.startswith("wall_z_pos_"):
        return "top"
    if label.startswith("wall_y_neg_") or label.startswith("wall_z_neg_"):
        return "bottom"
    if label.startswith("wall_x_pos_"):
        return "right"
    if label.startswith("wall_x_neg_"):
        return "left"
    return "other"


def _needs_rotation_for_side(width: float, height: float, side: str) -> bool:
    """Choose a simple, side-aware orientation for folded layout."""
    if side in ("top", "bottom"):
        return height > width
    if side in ("left", "right"):
        return width > height
    return False


def _dims_for(part: dict, rotated: bool) -> tuple[float, float]:
    if rotated:
        return part["height"], part["width"]
    return part["width"], part["height"]


def _layout_folded(parts: list[dict], padding: float, wall_offset: float) -> tuple[list[tuple], float, float]:
    """Place parts around bottom like an unfolded open box."""
    bottom_idx = next((i for i, p in enumerate(parts) if p["label"] == "bottom"), None)
    if bottom_idx is None:
        return _pack_parts(parts, padding)

    rotated_flags = [False] * len(parts)
    for i, part in enumerate(parts):
        if i == bottom_idx:
            continue
        side = _part_side(part["label"])
        rotated_flags[i] = _needs_rotation_for_side(part["width"], part["height"], side)

    groups = {"top": [], "bottom": [], "left": [], "right": [], "other": []}
    for i, part in enumerate(parts):
        if i == bottom_idx:
            continue
        side = _part_side(part["label"])
        if side in groups:
            groups[side].append(i)
        else:
            groups["other"].append(i)

    for side in groups:
        groups[side].sort(key=lambda idx: parts[idx]["label"])

    placements: dict[int, tuple[float, float]] = {bottom_idx: (0.0, 0.0)}
    bottom_w, bottom_h = _dims_for(parts[bottom_idx], rotated=False)

    def place_horizontal(indices: list[int], y_base: float, above: bool):
        if not indices:
            return
        total_w = sum(_dims_for(parts[i], rotated_flags[i])[0] for i in indices)
        total_w += padding * max(0, len(indices) - 1)
        x = (bottom_w - total_w) / 2.0
        for i in indices:
            w, h = _dims_for(parts[i], rotated_flags[i])
            y = y_base - h if above else y_base
            placements[i] = (x, y)
            x += w + padding

    def place_vertical(indices: list[int], x_base: float, left: bool):
        if not indices:
            return
        total_h = sum(_dims_for(parts[i], rotated_flags[i])[1] for i in indices)
        total_h += padding * max(0, len(indices) - 1)
        y = (bottom_h - total_h) / 2.0
        for i in indices:
            w, h = _dims_for(parts[i], rotated_flags[i])
            x = x_base - w if left else x_base
            placements[i] = (x, y)
            y += h + padding

    place_horizontal(groups["top"], y_base=-wall_offset, above=True)
    place_horizontal(groups["bottom"], y_base=bottom_h + wall_offset, above=False)
    place_vertical(groups["left"], x_base=-wall_offset, left=True)
    place_vertical(groups["right"], x_base=bottom_w + wall_offset, left=False)

    if groups["other"]:
        max_y = max(
            y + _dims_for(parts[i], rotated_flags[i])[1]
            for i, (x, y) in placements.items()
        )
        x = 0.0
        y = max_y + wall_offset
        for i in groups["other"]:
            w, h = _dims_for(parts[i], rotated_flags[i])
            placements[i] = (x, y)
            x += w + padding

    min_x = min(x for x, y in placements.values())
    min_y = min(y for x, y in placements.values())
    max_x = max(x + _dims_for(parts[i], rotated_flags[i])[0] for i, (x, y) in placements.items())
    max_y = max(y + _dims_for(parts[i], rotated_flags[i])[1] for i, (x, y) in placements.items())

    shift_x = padding - min_x
    shift_y = padding - min_y

    positions = [None] * len(parts)
    for i, (x, y) in placements.items():
        positions[i] = (x + shift_x, y + shift_y, rotated_flags[i])

    total_w = (max_x - min_x) + 2 * padding
    total_h = (max_y - min_y) + 2 * padding
    return positions, total_w, total_h


def _compute_layout(parts: list[dict], layout: str, padding: float, wall_offset: float) -> tuple[list[tuple], float, float]:
    if layout == "folded":
        return _layout_folded(parts, padding=padding, wall_offset=wall_offset)
    return _pack_parts(parts, padding)


def _transform_part(part: dict, ox: float, oy: float, rotated: bool):
    poly = part["polygon"]
    holes = part["holes"]
    w, h = part["width"], part["height"]

    if rotated:
        poly = _rotate_polygon_90(poly, w, h)
        holes = [_rotate_polygon_90(hole, w, h) for hole in holes]

    poly = [(x + ox, y + oy) for x, y in poly]
    holes = [[(x + ox, y + oy) for x, y in hole] for hole in holes]
    return poly, holes


def _add_face_to_dxf(
    msp,
    polygon: list[tuple[float, float]],
    layer: str,
    slot_cutouts: list[list[tuple[float, float]]] | None = None,
):
    """Add a single face to DXF modelspace."""
    if len(polygon) >= 2:
        pts = list(polygon)
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        msp.add_lwpolyline(pts, dxfattribs={"layer": layer})

    if slot_cutouts:
        for slot in slot_cutouts:
            if len(slot) >= 2:
                pts = list(slot)
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                msp.add_lwpolyline(pts, dxfattribs={"layer": layer})


def export_dxf(
    projections: dict[int, Projection2D],
    modified_polygons: dict[int, list[tuple[float, float]]],
    output_path: str,
    per_face: bool = False,
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] | None = None,
    layout: str = DEFAULT_LAYOUT,
    wall_offset: float = DEFAULT_FOLDED_OFFSET,
    padding: float = 5,
) -> list[str]:
    """Export faces to DXF file(s)."""
    import os

    written = []
    if slot_cutouts is None:
        slot_cutouts = {}

    if per_face:
        os.makedirs(output_path, exist_ok=True)
        for fid, polygon in modified_polygons.items():
            proj = projections[fid]
            filepath = os.path.join(output_path, f"{proj.label or f'face_{fid}'}.dxf")
            doc = ezdxf.new("R2010")
            msp = doc.modelspace()
            cutouts = slot_cutouts.get(fid, [])
            _add_face_to_dxf(msp, polygon, layer=proj.label or f"face_{fid}", slot_cutouts=cutouts)
            doc.saveas(filepath)
            written.append(filepath)
    else:
        if not output_path.endswith(".dxf"):
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, "lasercut.dxf")
        else:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        parts = _prepare_parts(projections, modified_polygons, slot_cutouts)
        positions, _, _ = _compute_layout(parts, layout=layout, padding=padding, wall_offset=wall_offset)

        layers_added = set()
        for part, pos in zip(parts, positions):
            if pos is None:
                continue
            layer_name = part["label"]
            if layer_name not in layers_added:
                doc.layers.add(layer_name, color=7)
                layers_added.add(layer_name)

            ox, oy, rotated = pos
            poly, holes = _transform_part(part, ox, oy, rotated)
            _add_face_to_dxf(msp, poly, layer=layer_name, slot_cutouts=holes)

        doc.saveas(output_path)
        written.append(output_path)

    return written


def export_svg(
    projections: dict[int, Projection2D],
    modified_polygons: dict[int, list[tuple[float, float]]],
    output_path: str,
    padding: float = 5,
    stroke_width: float = 0.5,
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] | None = None,
    layout: str = DEFAULT_LAYOUT,
    wall_offset: float = DEFAULT_FOLDED_OFFSET,
) -> str:
    """Export faces to SVG file with packed or folded layout."""
    import os

    if not output_path.endswith(".svg"):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "lasercut.svg")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    parts = _prepare_parts(projections, modified_polygons, slot_cutouts)
    positions, actual_w, actual_h = _compute_layout(parts, layout=layout, padding=padding, wall_offset=wall_offset)

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{actual_w:.1f}mm", f"{actual_h:.1f}mm"),
        viewBox=f"0 0 {actual_w:.1f} {actual_h:.1f}",
    )

    for part, pos in zip(parts, positions):
        if pos is None:
            continue
        ox, oy, rotated = pos
        group = dwg.g(id=part["label"])
        poly, holes = _transform_part(part, ox, oy, rotated)

        if len(poly) >= 2:
            group.add(
                dwg.polygon(
                    poly,
                    fill="none",
                    stroke="black",
                    stroke_width=stroke_width,
                )
            )

        for hole in holes:
            if len(hole) >= 2:
                group.add(
                    dwg.polygon(
                        hole,
                        fill="none",
                        stroke="black",
                        stroke_width=stroke_width,
                    )
                )

        _, label_h = _dims_for(part, rotated)
        group.add(
            dwg.text(
                part["label"],
                insert=(ox + 2, oy + label_h + 4),
                font_size="3",
                font_family="monospace",
                fill="blue",
            )
        )
        dwg.add(group)

    dwg.save()
    return output_path
