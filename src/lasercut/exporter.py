"""DXF and SVG export for laser cutting."""

import math
import ezdxf
import svgwrite
from .projector import Projection2D


def _bbox(polygon, inners=None):
    """Compute bounding box of a polygon plus optional inner polygons."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    if inners:
        for inner in inners:
            xs.extend(p[0] for p in inner)
            ys.extend(p[1] for p in inner)
    return min(xs), min(ys), max(xs), max(ys)


def _pack_parts(parts: list[dict], padding: float = 5) -> tuple[list[tuple], float, float]:
    """Pack parts efficiently using a bottom-left placement algorithm.

    Tries each part in both orientations (original and 90-degree rotated)
    and places it in the position that wastes the least vertical space.

    Returns:
        (positions, total_width, total_height) where positions is
        [(x_offset, y_offset, rotated), ...]
    """
    # Sort parts by height descending (tall pieces first = better packing)
    indexed_parts = list(enumerate(parts))
    indexed_parts.sort(key=lambda ip: max(ip[1]['width'], ip[1]['height']), reverse=True)

    placed = []  # (x, y, w, h, original_index, rotated)

    for orig_idx, part in indexed_parts:
        w0, h0 = part['width'], part['height']

        best_pos = None
        best_y_max = float('inf')
        best_rotated = False

        # Try both orientations
        for rotated, (w, h) in [(False, (w0, h0)), (True, (h0, w0))]:
            # Try placing at various x positions (left edges of existing parts + origin)
            x_candidates = [0] + [p[0] + p[2] + padding for p in placed]

            for x in x_candidates:
                # Find the lowest y where this part doesn't overlap with any placed part
                y = 0
                for px, py, pw, ph, _, _ in placed:
                    # Check horizontal overlap
                    if x < px + pw + padding and x + w + padding > px:
                        # There's horizontal overlap, so y must be above this part
                        y = max(y, py + ph + padding)

                y_max = y + h
                if y_max < best_y_max:
                    best_y_max = y_max
                    best_pos = (x, y, w, h)
                    best_rotated = rotated

        if best_pos:
            x, y, w, h = best_pos
            placed.append((x, y, w, h, orig_idx, best_rotated))

    # Convert back to original order
    positions = [None] * len(parts)
    for x, y, w, h, orig_idx, rotated in placed:
        positions[orig_idx] = (x + padding, y + padding, rotated)

    total_w = max((x + w for x, y, w, h, _, _ in placed), default=0) + 2 * padding
    total_h = max((y + h for x, y, w, h, _, _ in placed), default=0) + 2 * padding

    return positions, total_w, total_h


def _rotate_polygon_90(polygon, w, h):
    """Rotate a polygon 90 degrees clockwise. (x,y) -> (y, w-x)."""
    return [(y, w - x) for x, y in polygon]


def export_dxf(
    projections: dict[int, Projection2D],
    modified_polygons: dict[int, list[tuple[float, float]]],
    output_path: str,
    per_face: bool = False,
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] | None = None,
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
            _add_face_to_dxf(msp, proj, polygon, layer=proj.label or f"face_{fid}",
                             slot_cutouts=cutouts)
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

        for fid, polygon in modified_polygons.items():
            proj = projections[fid]
            layer_name = proj.label or f"face_{fid}"
            doc.layers.add(layer_name, color=7)
            cutouts = slot_cutouts.get(fid, [])
            _add_face_to_dxf(msp, proj, polygon, layer=layer_name,
                             slot_cutouts=cutouts)

        doc.saveas(output_path)
        written.append(output_path)

    return written


def _add_face_to_dxf(msp, proj: Projection2D, polygon: list[tuple[float, float]],
                     layer: str, slot_cutouts: list[list[tuple[float, float]]] | None = None):
    """Add a single face to DXF modelspace."""
    if len(polygon) >= 2:
        pts = list(polygon)
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        msp.add_lwpolyline(pts, dxfattribs={"layer": layer})

    # Add inner polygons (holes / through-slot cutouts from Shapely boolean ops)
    if slot_cutouts:
        for slot in slot_cutouts:
            if len(slot) >= 2:
                pts = list(slot)
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                msp.add_lwpolyline(pts, dxfattribs={"layer": layer})


def export_svg(
    projections: dict[int, Projection2D],
    modified_polygons: dict[int, list[tuple[float, float]]],
    output_path: str,
    padding: float = 5,
    stroke_width: float = 0.5,
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] | None = None,
) -> str:
    """Export faces to SVG file with optimized packing layout."""
    import os
    if not output_path.endswith(".svg"):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "lasercut.svg")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if slot_cutouts is None:
        slot_cutouts = {}

    # Prepare parts with bounding boxes
    parts = []
    for fid in sorted(modified_polygons.keys()):
        polygon = modified_polygons[fid]
        proj = projections[fid]

        if not polygon:
            continue

        # Collect all geometry for bounding box (outer polygon + holes from Shapely)
        all_polys = [polygon] + slot_cutouts.get(fid, [])
        xs = []
        ys = []
        for poly in all_polys:
            xs.extend(p[0] for p in poly)
            ys.extend(p[1] for p in poly)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max_x - min_x
        h = max_y - min_y

        # Normalize to origin
        norm_poly = [(x - min_x, y - min_y) for x, y in polygon]
        norm_holes = [
            [(x - min_x, y - min_y) for x, y in hole]
            for hole in slot_cutouts.get(fid, [])
        ]

        parts.append({
            'fid': fid,
            'label': proj.label or f"face_{fid}",
            'polygon': norm_poly,
            'holes': norm_holes,
            'width': w,
            'height': h,
        })

    # Pack parts efficiently
    positions, actual_w, actual_h = _pack_parts(parts, padding)

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{actual_w:.1f}mm", f"{actual_h:.1f}mm"),
        viewBox=f"0 0 {actual_w:.1f} {actual_h:.1f}",
    )

    for part, pos in zip(parts, positions):
        if pos is None:
            continue
        ox, oy, rotated = pos
        group = dwg.g(id=part['label'])

        poly = part['polygon']
        holes = part['holes']
        w, h = part['width'], part['height']

        if rotated:
            poly = _rotate_polygon_90(poly, w, h)
            holes = [_rotate_polygon_90(hole, w, h) for hole in holes]

        # Outer polygon
        pts = [(x + ox, y + oy) for x, y in poly]
        if len(pts) >= 2:
            group.add(dwg.polygon(
                pts,
                fill="none",
                stroke="black",
                stroke_width=stroke_width,
            ))

        # Inner polygons (holes / through-slots from Shapely)
        for hole in holes:
            hole_pts = [(x + ox, y + oy) for x, y in hole]
            if len(hole_pts) >= 2:
                group.add(dwg.polygon(
                    hole_pts,
                    fill="none",
                    stroke="black",
                    stroke_width=stroke_width,
                ))

        # Label
        lw = h if rotated else w
        lh = w if rotated else h
        group.add(dwg.text(
            part['label'],
            insert=(ox + 2, oy + lh + 4),
            font_size="3",
            font_family="monospace",
            fill="blue",
        ))

        dwg.add(group)

    dwg.save()
    return output_path
