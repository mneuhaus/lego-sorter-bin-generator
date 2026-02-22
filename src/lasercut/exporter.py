"""DXF and SVG export for laser cutting."""

import math

import ezdxf
import svgwrite

from .face_classifier import SharedEdge
from .projector import Projection2D, _project_point
from .step_loader import PlanarFace

DEFAULT_LAYOUT = "folded"
DEFAULT_FOLDED_OFFSET = 10.0


def _dist_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist_3d(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _points_close_3d(p1: tuple[float, float, float], p2: tuple[float, float, float], tol: float = 0.5) -> bool:
    return _dist_3d(p1, p2) < tol


def _pack_parts(parts: list[dict], padding: float = 5) -> tuple[list[tuple], float, float]:
    """Pack parts efficiently using a bottom-left placement algorithm."""
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
                "min_x": min_x,
                "min_y": min_y,
            }
        )

    return parts


def _triangle_area2(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _invert_3x3(m: list[list[float]]) -> list[list[float]] | None:
    """Invert a 3x3 matrix, returning None if singular."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]

    det = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(det) < 1e-12:
        return None

    inv_det = 1.0 / det
    return [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ]


def _mul_mat3_vec3(m: list[list[float]], v: list[float]) -> list[float]:
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def _solve_affine_from_corresponding_polygons(
    src_poly: list[tuple[float, float]],
    dst_poly: list[tuple[float, float]],
) -> tuple[float, float, float, float, float, float] | None:
    """Solve affine transform mapping corresponding source polygon points to destination."""
    n = min(len(src_poly), len(dst_poly))
    if n < 3:
        return None

    tri = None
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                if abs(_triangle_area2(src_poly[i], src_poly[j], src_poly[k])) > 1e-9:
                    tri = (i, j, k)
                    break
            if tri is not None:
                break
        if tri is not None:
            break
    if tri is None:
        return None

    i, j, k = tri
    m = [
        [src_poly[i][0], src_poly[i][1], 1.0],
        [src_poly[j][0], src_poly[j][1], 1.0],
        [src_poly[k][0], src_poly[k][1], 1.0],
    ]
    inv = _invert_3x3(m)
    if inv is None:
        return None

    vx = [dst_poly[i][0], dst_poly[j][0], dst_poly[k][0]]
    vy = [dst_poly[i][1], dst_poly[j][1], dst_poly[k][1]]
    a, b, tx = _mul_mat3_vec3(inv, vx)
    c, d, ty = _mul_mat3_vec3(inv, vy)
    return (a, b, c, d, tx, ty)


def _apply_affine(
    points: list[tuple[float, float]],
    affine: tuple[float, float, float, float, float, float],
) -> list[tuple[float, float]]:
    a, b, c, d, tx, ty = affine
    return [(a * x + b * y + tx, c * x + d * y + ty) for x, y in points]


def _find_matching_edge_index(proj: Projection2D, shared_edge: SharedEdge) -> int | None:
    """Find which 2D edge in the projection corresponds to the shared 3D edge."""
    tol = 0.5
    for idx, edge_3d in enumerate(proj.edge_map_3d):
        for se in [shared_edge.edge_a, shared_edge.edge_b]:
            if _points_close_3d(edge_3d.midpoint, se.midpoint, tol):
                return idx
            fwd = _points_close_3d(edge_3d.start, se.start, tol) and _points_close_3d(edge_3d.end, se.end, tol)
            rev = _points_close_3d(edge_3d.start, se.end, tol) and _points_close_3d(edge_3d.end, se.start, tol)
            if fwd or rev:
                return idx
    return None


def _edges_reversed(
    proj_a: Projection2D,
    edge_idx_a: int,
    proj_b: Projection2D,
    edge_idx_b: int,
    tol: float = 0.5,
) -> bool:
    """Check if two 2D edges correspond to reversed 3D edges."""
    ea = proj_a.edge_map_3d[edge_idx_a]
    eb = proj_b.edge_map_3d[edge_idx_b]

    fwd = _points_close_3d(ea.start, eb.start, tol) and _points_close_3d(ea.end, eb.end, tol)
    rev = _points_close_3d(ea.start, eb.end, tol) and _points_close_3d(ea.end, eb.start, tol)
    return rev and not fwd


def _outward_direction(p1: tuple[float, float], p2: tuple[float, float], polygon: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute outward normal from edge p1->p2 using polygon centroid heuristic."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ln = math.hypot(dx, dy)
    if ln < 1e-12:
        return (0.0, -1.0)

    n1 = (-dy / ln, dx / ln)
    n2 = (dy / ln, -dx / ln)

    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    mx = (p1[0] + p2[0]) * 0.5
    my = (p1[1] + p2[1]) * 0.5

    t1 = (mx + n1[0], my + n1[1])
    t2 = (mx + n2[0], my + n2[1])
    d1 = (t1[0] - cx) ** 2 + (t1[1] - cy) ** 2
    d2 = (t2[0] - cx) ** 2 + (t2[1] - cy) ** 2
    return n1 if d1 > d2 else n2


def _pt_to_local(part: dict, pt: tuple[float, float]) -> tuple[float, float]:
    return (pt[0] - part["min_x"], pt[1] - part["min_y"])


def _transform_points_to_edge(
    points: list[tuple[float, float]],
    src_a: tuple[float, float],
    src_b: tuple[float, float],
    dst_a: tuple[float, float],
    dst_b: tuple[float, float],
) -> list[tuple[float, float]]:
    """Rigid transform mapping src edge to dst edge."""
    v_src = (src_b[0] - src_a[0], src_b[1] - src_a[1])
    v_dst = (dst_b[0] - dst_a[0], dst_b[1] - dst_a[1])

    src_len = math.hypot(v_src[0], v_src[1])
    dst_len = math.hypot(v_dst[0], v_dst[1])
    if src_len < 1e-9 or dst_len < 1e-9:
        tx = dst_a[0] - src_a[0]
        ty = dst_a[1] - src_a[1]
        return [(x + tx, y + ty) for x, y in points]

    ang_src = math.atan2(v_src[1], v_src[0])
    ang_dst = math.atan2(v_dst[1], v_dst[0])
    ang = ang_dst - ang_src
    c = math.cos(ang)
    s = math.sin(ang)

    out = []
    for x, y in points:
        rx = c * (x - src_a[0]) - s * (y - src_a[1]) + dst_a[0]
        ry = s * (x - src_a[0]) + c * (y - src_a[1]) + dst_a[1]
        out.append((rx, ry))
    return out


def _reflect_points_across_line(
    points: list[tuple[float, float]],
    a: tuple[float, float],
    b: tuple[float, float],
) -> list[tuple[float, float]]:
    """Reflect points across infinite line through a->b."""
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    ln = math.hypot(vx, vy)
    if ln < 1e-12:
        return list(points)
    ux = vx / ln
    uy = vy / ln

    out = []
    for x, y in points:
        rx = x - a[0]
        ry = y - a[1]
        dot = rx * ux + ry * uy
        px = dot * ux
        py = dot * uy
        # reflection: keep along-line component, invert normal component
        fx = 2 * px - rx
        fy = 2 * py - ry
        out.append((a[0] + fx, a[1] + fy))
    return out


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _transform_part_packed(part: dict, ox: float, oy: float, rotated: bool):
    poly = part["polygon"]
    holes = part["holes"]
    w, h = part["width"], part["height"]
    if rotated:
        poly = _rotate_polygon_90(poly, w, h)
        holes = [_rotate_polygon_90(hole, w, h) for hole in holes]
    poly = [(x + ox, y + oy) for x, y in poly]
    holes = [[(x + ox, y + oy) for x, y in hole] for hole in holes]
    return poly, holes


def _build_bottom_anchors(
    projections: dict[int, Projection2D],
    shared_edges: list[SharedEdge] | None,
    bottom_id: int | None,
    faces: list[PlanarFace] | None,
) -> dict[int, dict]:
    """Build edge-anchoring data from bottom-to-wall joints."""
    anchors: dict[int, dict] = {}
    if bottom_id is None or shared_edges is None or bottom_id not in projections:
        return anchors

    bottom_proj = projections[bottom_id]

    # Direct bottom shared edges
    for se in shared_edges:
        if se.face_a_id != bottom_id and se.face_b_id != bottom_id:
            continue
        wall_id = se.face_b_id if se.face_a_id == bottom_id else se.face_a_id
        if wall_id not in projections:
            continue

        wall_proj = projections[wall_id]
        bottom_idx = _find_matching_edge_index(bottom_proj, se)
        wall_idx = _find_matching_edge_index(wall_proj, se)
        if bottom_idx is None or wall_idx is None:
            continue

        bp1, bp2 = bottom_proj.outer_edges_2d[bottom_idx]
        wp1, wp2 = wall_proj.outer_edges_2d[wall_idx]
        reversed_edge = _edges_reversed(bottom_proj, bottom_idx, wall_proj, wall_idx)

        anchors[wall_id] = {
            "bottom_edge": (bp1, bp2),
            "wall_edge": (wp1, wp2),
            "reversed": reversed_edge,
        }

    # Through-slot walls not directly adjacent to bottom
    if faces is not None:
        from .finger_joints import _find_bottom_edge_endpoints

        face_map = {f.face_id: f for f in faces}
        bottom_face = face_map.get(bottom_id)
        if bottom_face is not None:
            bottom_adjacent = set()
            for se in shared_edges:
                if se.face_a_id == bottom_id:
                    bottom_adjacent.add(se.face_b_id)
                elif se.face_b_id == bottom_id:
                    bottom_adjacent.add(se.face_a_id)

            for fid, wall_proj in projections.items():
                if fid == bottom_id or fid in anchors or fid in bottom_adjacent:
                    continue
                wall_face = face_map.get(fid)
                if wall_face is None:
                    continue

                endpoints = _find_bottom_edge_endpoints(wall_face, bottom_face)
                if endpoints is None:
                    continue
                p_start_3d, p_end_3d = endpoints

                bp1 = _project_point(p_start_3d, bottom_proj.origin_3d, bottom_proj.u_axis, bottom_proj.v_axis)
                bp2 = _project_point(p_end_3d, bottom_proj.origin_3d, bottom_proj.u_axis, bottom_proj.v_axis)
                wp1 = _project_point(p_start_3d, wall_proj.origin_3d, wall_proj.u_axis, wall_proj.v_axis)
                wp2 = _project_point(p_end_3d, wall_proj.origin_3d, wall_proj.u_axis, wall_proj.v_axis)

                anchors[fid] = {
                    "bottom_edge": (bp1, bp2),
                    "wall_edge": (wp1, wp2),
                    "reversed": False,
                }

    return anchors


def _arrange_packed(parts: list[dict], padding: float) -> tuple[list[dict], float, float]:
    positions, total_w, total_h = _pack_parts(parts, padding)
    geoms = []
    for part, pos in zip(parts, positions):
        if pos is None:
            continue
        ox, oy, rotated = pos
        poly, holes = _transform_part_packed(part, ox, oy, rotated)
        geoms.append({"fid": part["fid"], "label": part["label"], "polygon": poly, "holes": holes})
    return geoms, total_w, total_h


def _shift_geoms_to_positive(geoms: list[dict], padding: float) -> tuple[list[dict], float, float]:
    xs = []
    ys = []
    for g in geoms:
        for x, y in g["polygon"]:
            xs.append(x)
            ys.append(y)
        for hole in g["holes"]:
            for x, y in hole:
                xs.append(x)
                ys.append(y)

    if not xs:
        return geoms, 2 * padding, 2 * padding

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = padding - min_x
    dy = padding - min_y

    shifted = []
    for g in geoms:
        poly = [(x + dx, y + dy) for x, y in g["polygon"]]
        holes = [[(x + dx, y + dy) for x, y in hole] for hole in g["holes"]]
        shifted.append({"fid": g["fid"], "label": g["label"], "polygon": poly, "holes": holes})

    total_w = (max_x - min_x) + 2 * padding
    total_h = (max_y - min_y) + 2 * padding
    return shifted, total_w, total_h


def _arrange_folded(
    parts: list[dict],
    projections: dict[int, Projection2D],
    shared_edges: list[SharedEdge] | None,
    bottom_id: int | None,
    faces: list[PlanarFace] | None,
    wall_offset: float,
    padding: float,
) -> tuple[list[dict], float, float]:
    part_by_fid = {p["fid"]: p for p in parts}
    if bottom_id is None or bottom_id not in part_by_fid:
        return _arrange_packed(parts, padding)

    bottom_part = part_by_fid[bottom_id]
    anchors = _build_bottom_anchors(projections, shared_edges, bottom_id, faces)
    if not anchors:
        return _arrange_packed(parts, padding)

    geoms = []
    geoms.append(
        {
            "fid": bottom_part["fid"],
            "label": bottom_part["label"],
            "polygon": list(bottom_part["polygon"]),
            "holes": [list(h) for h in bottom_part["holes"]],
        }
    )
    placed = {bottom_id}

    bottom_poly = bottom_part["polygon"]

    for fid, anchor in anchors.items():
        part = part_by_fid.get(fid)
        if part is None:
            continue

        bp1, bp2 = anchor["bottom_edge"]
        wp1, wp2 = anchor["wall_edge"]

        bp1_local = _pt_to_local(bottom_part, bp1)
        bp2_local = _pt_to_local(bottom_part, bp2)
        wp1_local = _pt_to_local(part, wp1)
        wp2_local = _pt_to_local(part, wp2)

        outward = _outward_direction(bp1_local, bp2_local, bottom_poly)
        tp1 = (bp1_local[0] + outward[0] * wall_offset, bp1_local[1] + outward[1] * wall_offset)
        tp2 = (bp2_local[0] + outward[0] * wall_offset, bp2_local[1] + outward[1] * wall_offset)

        if anchor["reversed"]:
            tp1, tp2 = tp2, tp1

        poly = _transform_points_to_edge(part["polygon"], wp1_local, wp2_local, tp1, tp2)
        holes = [_transform_points_to_edge(h, wp1_local, wp2_local, tp1, tp2) for h in part["holes"]]

        # If the transformed wall ended up on the inner side of the bottom edge,
        # mirror it across the edge. Per-face projection bases are not guaranteed
        # to have consistent handedness, so rotation-only alignment is not enough.
        c = _centroid(poly)
        m = ((tp1[0] + tp2[0]) * 0.5, (tp1[1] + tp2[1]) * 0.5)
        side = (c[0] - m[0]) * outward[0] + (c[1] - m[1]) * outward[1]
        if side < 0:
            poly = _reflect_points_across_line(poly, tp1, tp2)
            holes = [_reflect_points_across_line(h, tp1, tp2) for h in holes]

        geoms.append({"fid": part["fid"], "label": part["label"], "polygon": poly, "holes": holes})
        placed.add(fid)

    # Any leftover parts get packed below the unfolded cluster.
    leftovers = [p for p in parts if p["fid"] not in placed]
    if leftovers:
        packed, _, _ = _arrange_packed(leftovers, padding=padding)
        cur_max_y = max(max(y for _, y in g["polygon"]) for g in geoms)
        for g in packed:
            poly = [(x, y + cur_max_y + wall_offset) for x, y in g["polygon"]]
            holes = [[(x, y + cur_max_y + wall_offset) for x, y in hole] for hole in g["holes"]]
            geoms.append({"fid": g["fid"], "label": g["label"], "polygon": poly, "holes": holes})

    return _shift_geoms_to_positive(geoms, padding)


def _compute_arrangement(
    parts: list[dict],
    layout: str,
    projections: dict[int, Projection2D],
    shared_edges: list[SharedEdge] | None,
    bottom_id: int | None,
    faces: list[PlanarFace] | None,
    padding: float,
    wall_offset: float,
) -> tuple[list[dict], float, float]:
    if layout == "folded":
        return _arrange_folded(
            parts,
            projections=projections,
            shared_edges=shared_edges,
            bottom_id=bottom_id,
            faces=faces,
            wall_offset=wall_offset,
            padding=padding,
        )
    return _arrange_packed(parts, padding)


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
    shared_edges: list[SharedEdge] | None = None,
    bottom_id: int | None = None,
    faces: list[PlanarFace] | None = None,
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
        geoms, _, _ = _compute_arrangement(
            parts,
            layout=layout,
            projections=projections,
            shared_edges=shared_edges,
            bottom_id=bottom_id,
            faces=faces,
            padding=padding,
            wall_offset=wall_offset,
        )

        layers_added = set()
        for g in geoms:
            layer_name = g["label"]
            if layer_name not in layers_added:
                doc.layers.add(layer_name, color=7)
                layers_added.add(layer_name)
            _add_face_to_dxf(msp, g["polygon"], layer=layer_name, slot_cutouts=g["holes"])

        doc.saveas(output_path)
        written.append(output_path)

    return written


def _add_lego_engravings(dwg, group, polygon, fid, bottom_id):
    """Add decorative Lego-themed engravings inside a face polygon."""
    from shapely.geometry import Polygon as SPoly, Point as SPoint

    if len(polygon) < 3:
        return

    poly = SPoly(polygon)
    if poly.is_empty or not poly.is_valid:
        return

    bounds = poly.bounds  # (minx, miny, maxx, maxy)
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2

    engrave_color = "red"
    engrave_width = 0.3

    is_bottom = fid == bottom_id

    if is_bottom:
        # Bottom plate: grid of Lego studs
        stud_r = 4.8 / 2  # classic stud radius
        spacing = 8.0  # standard Lego pitch
        inner_r = stud_r * 0.55

        # Offset grid to center on the polygon
        ox = bounds[0] + ((w % spacing) / 2) + spacing / 2
        oy = bounds[1] + ((h % spacing) / 2) + spacing / 2

        y = oy
        while y < bounds[3] - spacing / 4:
            x = ox
            while x < bounds[2] - spacing / 4:
                pt = SPoint(x, y)
                if poly.contains(pt) and poly.boundary.distance(pt) > stud_r + 1.5:
                    group.add(dwg.circle(
                        center=(x, y), r=stud_r,
                        fill="none", stroke=engrave_color,
                        stroke_width=engrave_width,
                    ))
                    group.add(dwg.circle(
                        center=(x, y), r=inner_r,
                        fill="none", stroke=engrave_color,
                        stroke_width=engrave_width * 0.6,
                    ))
                x += spacing
            y += spacing
    else:
        # Walls: brick pattern
        brick_h = 8.0
        brick_w = 16.0
        margin = 6.0

        # Inset polygon for pattern boundary
        inset = poly.buffer(-margin)
        if inset.is_empty:
            return
        from shapely.geometry import MultiPolygon as MPolygon
        if isinstance(inset, MPolygon):
            inset = max(inset.geoms, key=lambda g: g.area)

        ib = inset.bounds
        iw = ib[2] - ib[0]
        ih = ib[3] - ib[1]

        if iw < brick_w or ih < brick_h * 1.5:
            return

        # Draw horizontal mortar lines
        row = 0
        y = ib[1] + brick_h
        while y < ib[3] - 1:
            from shapely.geometry import LineString
            line = LineString([(ib[0], y), (ib[2], y)])
            clipped = line.intersection(inset)
            if not clipped.is_empty:
                if clipped.geom_type == "LineString":
                    coords = list(clipped.coords)
                    if len(coords) >= 2:
                        group.add(dwg.line(
                            start=coords[0], end=coords[-1],
                            stroke=engrave_color, stroke_width=engrave_width,
                        ))
                elif clipped.geom_type == "MultiLineString":
                    for seg in clipped.geoms:
                        coords = list(seg.coords)
                        if len(coords) >= 2:
                            group.add(dwg.line(
                                start=coords[0], end=coords[-1],
                                stroke=engrave_color, stroke_width=engrave_width,
                            ))

            # Draw vertical mortar lines (staggered)
            offset = (brick_w / 2) if (row % 2 == 1) else 0
            prev_y = y - brick_h
            x = ib[0] + offset + brick_w
            while x < ib[2] - 1:
                top = (x, prev_y)
                bot = (x, y)
                from shapely.geometry import LineString as LS
                vline = LS([top, bot])
                vclipped = vline.intersection(inset)
                if not vclipped.is_empty:
                    if vclipped.geom_type == "LineString":
                        vc = list(vclipped.coords)
                        if len(vc) >= 2:
                            group.add(dwg.line(
                                start=vc[0], end=vc[-1],
                                stroke=engrave_color, stroke_width=engrave_width,
                            ))
                    elif vclipped.geom_type == "MultiLineString":
                        for seg in vclipped.geoms:
                            vc = list(seg.coords)
                            if len(vc) >= 2:
                                group.add(dwg.line(
                                    start=vc[0], end=vc[-1],
                                    stroke=engrave_color, stroke_width=engrave_width,
                                ))
                x += brick_w

            row += 1
            y += brick_h

        # Top row vertical lines (from last horizontal line to top of inset)
        last_y = ib[1] + brick_h * row
        if last_y < ib[3]:
            offset = (brick_w / 2) if (row % 2 == 1) else 0
            x = ib[0] + offset + brick_w
            while x < ib[2] - 1:
                from shapely.geometry import LineString as LS2
                vline = LS2([(x, last_y), (x, ib[3])])
                vclipped = vline.intersection(inset)
                if not vclipped.is_empty and vclipped.geom_type == "LineString":
                    vc = list(vclipped.coords)
                    if len(vc) >= 2:
                        group.add(dwg.line(
                            start=vc[0], end=vc[-1],
                            stroke=engrave_color, stroke_width=engrave_width,
                        ))
                x += brick_w


def export_svg(
    projections: dict[int, Projection2D],
    modified_polygons: dict[int, list[tuple[float, float]]],
    output_path: str,
    padding: float = 5,
    stroke_width: float = 0.5,
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] | None = None,
    layout: str = DEFAULT_LAYOUT,
    wall_offset: float = DEFAULT_FOLDED_OFFSET,
    shared_edges: list[SharedEdge] | None = None,
    bottom_id: int | None = None,
    faces: list[PlanarFace] | None = None,
    overlay_original: bool = False,
) -> str:
    """Export faces to SVG file with packed or edge-anchored folded layout."""
    import os

    if not output_path.endswith(".svg"):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "lasercut.svg")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    parts = _prepare_parts(projections, modified_polygons, slot_cutouts)
    part_by_fid = {p["fid"]: p for p in parts}
    geoms, actual_w, actual_h = _compute_arrangement(
        parts,
        layout=layout,
        projections=projections,
        shared_edges=shared_edges,
        bottom_id=bottom_id,
        faces=faces,
        padding=padding,
        wall_offset=wall_offset,
    )

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{actual_w:.1f}mm", f"{actual_h:.1f}mm"),
        viewBox=f"0 0 {actual_w:.1f} {actual_h:.1f}",
    )

    for g in geoms:
        group = dwg.g(id=g["label"])

        if len(g["polygon"]) >= 2:
            group.add(
                dwg.polygon(
                    g["polygon"],
                    fill="none",
                    stroke="black",
                    stroke_width=stroke_width,
                )
            )

        for hole in g["holes"]:
            if len(hole) >= 2:
                group.add(
                    dwg.polygon(
                        hole,
                        fill="none",
                        stroke="black",
                        stroke_width=stroke_width,
                    )
                )

        if overlay_original:
            part = part_by_fid.get(g["fid"])
            proj = projections.get(g["fid"])
            if part is not None and proj is not None and len(part["polygon"]) >= 3 and len(g["polygon"]) >= 3:
                affine = _solve_affine_from_corresponding_polygons(part["polygon"], g["polygon"])
                if affine is not None and len(proj.outer_polygon) >= 3:
                    orig_outer_local = [(x - part["min_x"], y - part["min_y"]) for x, y in proj.outer_polygon]
                    orig_outer = _apply_affine(orig_outer_local, affine)
                    group.add(
                        dwg.polygon(
                            orig_outer,
                            fill="none",
                            stroke="#00a651",
                            stroke_width=max(0.2, stroke_width * 0.8),
                            stroke_dasharray="2,2",
                        )
                    )

                    for inner in proj.inner_polygons:
                        if len(inner) < 3:
                            continue
                        orig_inner_local = [(x - part["min_x"], y - part["min_y"]) for x, y in inner]
                        orig_inner = _apply_affine(orig_inner_local, affine)
                        group.add(
                            dwg.polygon(
                                orig_inner,
                                fill="none",
                                stroke="#00a651",
                                stroke_width=max(0.2, stroke_width * 0.8),
                                stroke_dasharray="2,2",
                            )
                        )

        # Lego engravings (disabled for now)
        # _add_lego_engravings(dwg, group, g["polygon"], g.get("fid"), bottom_id)

        dwg.add(group)

    dwg.save()
    return output_path


def _shift_points(points: list[tuple[float, float]], dx: float, dy: float) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in points]


def _build_overlay_panel(
    part_a: dict,
    part_b: dict,
    a_edge: tuple[tuple[float, float], tuple[float, float]],
    b_edge: tuple[tuple[float, float], tuple[float, float]],
    label: str,
) -> dict | None:
    """Build one overlap-debug panel by aligning B's edge onto A's edge."""
    ap1_local = _pt_to_local(part_a, a_edge[0])
    ap2_local = _pt_to_local(part_a, a_edge[1])
    bp1_local = _pt_to_local(part_b, b_edge[0])
    bp2_local = _pt_to_local(part_b, b_edge[1])

    a_poly = list(part_a["polygon"])
    a_holes = [list(h) for h in part_a["holes"]]

    b_poly = _transform_points_to_edge(part_b["polygon"], bp1_local, bp2_local, ap1_local, ap2_local)
    b_holes = [_transform_points_to_edge(h, bp1_local, bp2_local, ap1_local, ap2_local) for h in part_b["holes"]]

    # Mirror B across the shared edge if needed so both face interiors are on
    # the same side, making mismatch regions visually obvious via alpha overlap.
    def _side(pt: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return _triangle_area2(p1, p2, pt)

    a_side = _side(_centroid(a_poly), ap1_local, ap2_local)
    b_side = _side(_centroid(b_poly), ap1_local, ap2_local)
    if a_side * b_side < 0:
        b_poly = _reflect_points_across_line(b_poly, ap1_local, ap2_local)
        b_holes = [_reflect_points_across_line(h, ap1_local, ap2_local) for h in b_holes]

    xs = [p[0] for p in a_poly] + [p[0] for p in b_poly]
    ys = [p[1] for p in a_poly] + [p[1] for p in b_poly]
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return {
        "label": label,
        "a_poly": a_poly,
        "a_holes": a_holes,
        "b_poly": b_poly,
        "b_holes": b_holes,
        "edge": (ap1_local, ap2_local),
        "min_x": min_x,
        "min_y": min_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def export_svg_overlap_debug(
    projections: dict[int, Projection2D],
    modified_polygons: dict[int, list[tuple[float, float]]],
    output_path: str,
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] | None = None,
    shared_edges: list[SharedEdge] | None = None,
    bottom_id: int | None = None,
    faces: list[PlanarFace] | None = None,
    panel_padding: float = 8.0,
    panel_gap: float = 12.0,
) -> str:
    """Export per-joint translucent overlap panels for visual mesh verification."""
    import os

    if not output_path.endswith(".svg"):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "lasercut-verify-overlap.svg")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    parts = _prepare_parts(projections, modified_polygons, slot_cutouts)
    part_by_fid = {p["fid"]: p for p in parts}
    panels: list[dict] = []
    direct_pair_keys: set[tuple[int, int]] = set()

    if shared_edges is not None:
        for i, se in enumerate(shared_edges):
            fid_a = se.face_a_id
            fid_b = se.face_b_id
            if fid_a not in projections or fid_b not in projections:
                continue
            part_a = part_by_fid.get(fid_a)
            part_b = part_by_fid.get(fid_b)
            if part_a is None or part_b is None:
                continue

            edge_idx_a = _find_matching_edge_index(projections[fid_a], se)
            edge_idx_b = _find_matching_edge_index(projections[fid_b], se)
            if edge_idx_a is None or edge_idx_b is None:
                continue

            a_edge = projections[fid_a].outer_edges_2d[edge_idx_a]
            b_edge = projections[fid_b].outer_edges_2d[edge_idx_b]
            panel = _build_overlay_panel(
                part_a,
                part_b,
                a_edge,
                b_edge,
                label=f"shared-{i}: {part_a['label']} vs {part_b['label']}",
            )
            if panel is not None:
                panels.append(panel)
                direct_pair_keys.add(tuple(sorted((fid_a, fid_b))))

    # Add through-slot bottom-wall overlays (including extra wall instances).
    if bottom_id is not None and faces is not None and shared_edges is not None and bottom_id in part_by_fid:
        anchors = _build_bottom_anchors(
            projections=projections,
            shared_edges=shared_edges,
            bottom_id=bottom_id,
            faces=faces,
        )
        bottom_part = part_by_fid[bottom_id]
        for wall_id, anchor in anchors.items():
            wall_part = part_by_fid.get(wall_id)
            if wall_part is None:
                continue
            pair_key = tuple(sorted((bottom_id, wall_id)))
            if pair_key in direct_pair_keys:
                continue

            bp1, bp2 = anchor["bottom_edge"]
            wp1, wp2 = anchor["wall_edge"]
            if anchor.get("reversed"):
                wp1, wp2 = wp2, wp1
            panel = _build_overlay_panel(
                bottom_part,
                wall_part,
                (bp1, bp2),
                (wp1, wp2),
                label=f"through: {bottom_part['label']} vs {wall_part['label']}",
            )
            if panel is not None:
                panels.append(panel)

    if not panels:
        dwg = svgwrite.Drawing(
            output_path,
            size=("200mm", "60mm"),
            viewBox="0 0 200 60",
        )
        dwg.add(dwg.text("No overlap panels found", insert=(8, 24), fill="black", font_size="8px"))
        dwg.save()
        return output_path

    cursor_y = panel_padding
    max_panel_w = 0.0
    for panel in panels:
        dx = panel_padding - panel["min_x"]
        dy = cursor_y - panel["min_y"]
        panel["a_poly"] = _shift_points(panel["a_poly"], dx, dy)
        panel["a_holes"] = [_shift_points(h, dx, dy) for h in panel["a_holes"]]
        panel["b_poly"] = _shift_points(panel["b_poly"], dx, dy)
        panel["b_holes"] = [_shift_points(h, dx, dy) for h in panel["b_holes"]]
        panel["edge"] = (
            (panel["edge"][0][0] + dx, panel["edge"][0][1] + dy),
            (panel["edge"][1][0] + dx, panel["edge"][1][1] + dy),
        )
        cursor_y += panel["height"] + panel_gap
        max_panel_w = max(max_panel_w, panel["width"])

    canvas_w = max_panel_w + 2 * panel_padding
    canvas_h = cursor_y - panel_gap + panel_padding
    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{canvas_w:.1f}mm", f"{canvas_h:.1f}mm"),
        viewBox=f"0 0 {canvas_w:.1f} {canvas_h:.1f}",
    )

    for panel in panels:
        group = dwg.g(id=panel["label"].replace(" ", "_"))
        group.add(
            dwg.polygon(
                panel["a_poly"],
                fill="#1f77b4",
                fill_opacity=0.5,
                stroke="#1f77b4",
                stroke_width=0.35,
            )
        )
        for hole in panel["a_holes"]:
            group.add(
                dwg.polygon(
                    hole,
                    fill="white",
                    fill_opacity=1.0,
                    stroke="#1f77b4",
                    stroke_width=0.25,
                )
            )

        group.add(
            dwg.polygon(
                panel["b_poly"],
                fill="#d62728",
                fill_opacity=0.5,
                stroke="#d62728",
                stroke_width=0.35,
            )
        )
        for hole in panel["b_holes"]:
            group.add(
                dwg.polygon(
                    hole,
                    fill="white",
                    fill_opacity=1.0,
                    stroke="#d62728",
                    stroke_width=0.25,
                )
            )

        group.add(
            dwg.line(
                start=panel["edge"][0],
                end=panel["edge"][1],
                stroke="black",
                stroke_width=0.45,
                stroke_dasharray="1,1",
            )
        )
        text_pos = (panel["edge"][0][0] + 2.0, min(panel["edge"][0][1], panel["edge"][1][1]) - 1.5)
        group.add(
            dwg.text(
                panel["label"],
                insert=text_pos,
                fill="black",
                font_size="4px",
            )
        )
        dwg.add(group)

    dwg.save()
    return output_path
