"""Project 3D panel faces to 2D outlines and export as SVG.

Panels are arranged in an "unfolded" layout -- as if the 3D box were opened
flat.  Adjacent panels are placed next to their shared edges with a 4 mm gap
so that finger-joint alignment can be visually verified.
"""

from __future__ import annotations

import math
from itertools import permutations
from collections import deque
from dataclasses import dataclass

import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepTools import BRepTools_WireExplorer

from lasercut.panels import BinModel, SharedEdge, _vec_cross, _vec_len, _vec_dot

try:
    import svgwrite
except ImportError:
    svgwrite = None  # type: ignore[assignment]

try:
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
except ImportError:
    Polygon = None  # type: ignore[assignment]
    LineString = None  # type: ignore[assignment]
    unary_union = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 2D helpers
# ---------------------------------------------------------------------------

def _rotate_pt(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (x * c - y * s, x * s + y * c)


def _rotate_pts(
    pts: list[tuple[float, float]], angle: float,
) -> list[tuple[float, float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [(x * c - y * s, x * s + y * c) for x, y in pts]


def _translate_pts(
    pts: list[tuple[float, float]], dx: float, dy: float,
) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in pts]


def _collapse_short_segments(
    pts: list[tuple[float, float]],
    min_len: float = 0.08,
) -> list[tuple[float, float]]:
    """Remove tiny colinear edge fragments from a closed polyline.

    This intentionally avoids collapsing short segments at corners, because
    those can encode real right-angle geometry and collapsing them can create
    visible diagonal artifacts.
    """
    if len(pts) < 4:
        return pts

    def _is_colinear(
        v1: tuple[float, float],
        v2: tuple[float, float],
        cos_tol: float = 0.995,
    ) -> bool:
        l1 = math.hypot(v1[0], v1[1])
        l2 = math.hypot(v2[0], v2[1])
        if l1 < 1e-9 or l2 < 1e-9:
            return False
        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
        return abs(dot) >= cos_tol

    out = list(pts)
    changed = True
    while changed and len(out) >= 4:
        changed = False
        n = len(out)
        for i in range(n):
            j = (i + 1) % n
            dx = out[j][0] - out[i][0]
            dy = out[j][1] - out[i][1]
            if math.hypot(dx, dy) >= min_len:
                continue
            prev_i = (i - 1) % n
            next_j = (j + 1) % n

            v_prev = (out[i][0] - out[prev_i][0], out[i][1] - out[prev_i][1])
            v_short = (out[j][0] - out[i][0], out[j][1] - out[i][1])
            v_next = (out[next_j][0] - out[j][0], out[next_j][1] - out[j][1])

            # Only collapse when this tiny edge is part of a nearly straight run.
            if not (_is_colinear(v_prev, v_short) and _is_colinear(v_short, v_next)):
                continue

            del out[j]
            changed = True
            break
    return out


def _filter_boundary_touching_holes(
    outline: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]],
    boundary_tol: float = 0.03,
) -> list[list[tuple[float, float]]]:
    """Drop degenerate inner loops that touch the outer boundary."""
    if not holes:
        return holes
    if Polygon is None:
        return holes
    try:
        outer_poly = Polygon(outline)
        if not outer_poly.is_valid or outer_poly.area <= 0:
            return holes
    except Exception:
        return holes

    kept: list[list[tuple[float, float]]] = []
    for hole in holes:
        if len(hole) < 3:
            continue
        try:
            hole_poly = Polygon(hole)
            if not hole_poly.is_valid or hole_poly.area <= 0:
                continue
            # Keep only strict interior loops with measurable clearance.
            if not outer_poly.contains(hole_poly):
                continue
            if outer_poly.exterior.distance(hole_poly) <= boundary_tol:
                continue
            kept.append(hole)
        except Exception:
            continue
    return kept


def _point_in_polygon(
    pt: tuple[float, float],
    poly: list[tuple[float, float]],
) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = pt
    inside = False
    n = len(poly)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)):
            x_inter = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_inter:
                inside = not inside
        j = i
    return inside


def _overlap_area(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> float:
    if Polygon is None:
        return 0.0
    try:
        pa = Polygon(a)
        pb = Polygon(b)
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        return pa.intersection(pb).area
    except Exception:
        return 0.0


def _distance_between(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> float:
    if Polygon is None:
        return float("inf")
    try:
        pa = Polygon(a)
        pb = Polygon(b)
        if not pa.is_valid or not pb.is_valid:
            return float("inf")
        return pa.distance(pb)
    except Exception:
        return float("inf")


def _any_overlap(
    outline: list[tuple[float, float]],
    others: list[list[tuple[float, float]]],
    min_area: float = 0.01,
) -> bool:
    for other in others:
        if _overlap_area(outline, other) > min_area:
            return True
    return False


def _total_overlap_area(
    outline: list[tuple[float, float]],
    others: list[list[tuple[float, float]]],
) -> float:
    return sum(_overlap_area(outline, other) for other in others)


def _too_close(
    outline: list[tuple[float, float]],
    others: list[list[tuple[float, float]]],
    min_clearance: float,
) -> bool:
    if Polygon is None:
        return False
    for other in others:
        if _distance_between(outline, other) < min_clearance:
            return True
    return False


def _push_out_until_clear(
    outline: list[tuple[float, float]],
    out_n: tuple[float, float],
    placed_others: list[list[tuple[float, float]]],
    step: float,
    min_clearance: float = 0.0,
    max_extra: float = 80.0,
) -> tuple[list[tuple[float, float]], float]:
    """Translate outline along out_n until it no longer overlaps placed panels."""
    needs_move = _any_overlap(outline, placed_others)
    if min_clearance > 0.0:
        needs_move = needs_move or _too_close(outline, placed_others, min_clearance)
    if not placed_others or not needs_move:
        return outline, 0.0

    def _push_dir(sign: float) -> tuple[list[tuple[float, float]], float, bool]:
        moved = list(outline)
        pushed = 0.0
        while pushed < max_extra:
            overlap = _any_overlap(moved, placed_others)
            close = min_clearance > 0.0 and _too_close(moved, placed_others, min_clearance)
            if not overlap and not close:
                break
            moved = _translate_pts(moved, out_n[0] * step * sign, out_n[1] * step * sign)
            pushed += step
        overlap = _any_overlap(moved, placed_others)
        close = min_clearance > 0.0 and _too_close(moved, placed_others, min_clearance)
        return moved, pushed * sign, not overlap and not close

    moved_fwd, pushed_fwd, ok_fwd = _push_dir(1.0)
    if ok_fwd:
        return moved_fwd, pushed_fwd

    moved_rev, pushed_rev, ok_rev = _push_dir(-1.0)
    if ok_rev:
        return moved_rev, pushed_rev

    # Neither direction fully cleared overlap; keep the better one.
    area_fwd = _total_overlap_area(moved_fwd, placed_others)
    area_rev = _total_overlap_area(moved_rev, placed_others)
    if area_rev < area_fwd:
        return moved_rev, pushed_rev
    return moved_fwd, pushed_fwd


def _bbox_area(pts: list[tuple[float, float]]) -> float:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _min_bbox_angle(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    angles: set[float] = {0.0}
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        if dx * dx + dy * dy < 1e-6:
            continue
        a = math.atan2(dy, dx) % (math.pi / 2)
        angles.add(a)
    best_angle = 0.0
    best_area = float("inf")
    for a in angles:
        rotated = _rotate_pts(pts, -a)
        area = _bbox_area(rotated)
        if area < best_area:
            best_area = area
            best_angle = a
    return best_angle


def _reflect_across_line(
    x: float, y: float, angle: float,
) -> tuple[float, float]:
    """Reflect point (x, y) across a line through the origin at *angle*."""
    c2 = math.cos(2 * angle)
    s2 = math.sin(2 * angle)
    return (c2 * x + s2 * y, s2 * x - c2 * y)


# ---------------------------------------------------------------------------
# Affine transform helper (rotation/reflection + translation)
# ---------------------------------------------------------------------------

@dataclass
class Affine2D:
    """2D affine transform: P' = M @ P + t.

    M is stored as (a, b, c, d) where M = [[a, b], [c, d]].
    """
    a: float
    b: float
    c: float
    d: float
    tx: float
    ty: float

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (self.a * x + self.b * y + self.tx,
                self.c * x + self.d * y + self.ty)

    def apply_pts(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [self.apply(x, y) for x, y in pts]

    @staticmethod
    def identity() -> "Affine2D":
        return Affine2D(1, 0, 0, 1, 0, 0)

    @staticmethod
    def from_rotation(angle: float) -> "Affine2D":
        c, s = math.cos(angle), math.sin(angle)
        return Affine2D(c, -s, s, c, 0, 0)

    @staticmethod
    def from_reflection(axis_angle: float) -> "Affine2D":
        """Reflection across a line through origin at axis_angle."""
        c2 = math.cos(2 * axis_angle)
        s2 = math.sin(2 * axis_angle)
        return Affine2D(c2, s2, s2, -c2, 0, 0)

    @staticmethod
    def from_translation(tx: float, ty: float) -> "Affine2D":
        return Affine2D(1, 0, 0, 1, tx, ty)

    def compose(self, other: "Affine2D") -> "Affine2D":
        """Return self(other(P)) = self.M @ (other.M @ P + other.t) + self.t."""
        a = self.a * other.a + self.b * other.c
        b = self.a * other.b + self.b * other.d
        c = self.c * other.a + self.d * other.c
        d = self.c * other.b + self.d * other.d
        tx = self.a * other.tx + self.b * other.ty + self.tx
        ty = self.c * other.tx + self.d * other.ty + self.ty
        return Affine2D(a, b, c, d, tx, ty)


# ---------------------------------------------------------------------------
# Panel2D projection with 3D -> 2D mapping
# ---------------------------------------------------------------------------

@dataclass
class Panel2D:
    name: str
    outline: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]]
    u_axis: tuple[float, float, float]
    v_axis: tuple[float, float, float]
    offset_x: float
    offset_y: float

    def project_3d(self, pt: tuple[float, float, float]) -> tuple[float, float]:
        """Map an arbitrary 3D point into this panel's local 2D coordinates."""
        return (
            _vec_dot(pt, self.u_axis) - self.offset_x,
            _vec_dot(pt, self.v_axis) - self.offset_y,
        )


@dataclass
class Piece2D:
    name: str
    outline: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]]


def _poly_centroid_xy(pts: list[tuple[float, float]]) -> tuple[float, float]:
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _ring_to_loop(ring) -> list[tuple[float, float]]:
    coords = list(ring.coords)
    if len(coords) <= 1:
        return []
    return [(float(x), float(y)) for x, y in coords[:-1]]


def _rect_loop(
    origin: tuple[float, float],
    x_dir: tuple[float, float],
    y_dir: tuple[float, float],
    dx: float,
    dy: float,
) -> list[tuple[float, float]]:
    x0, y0 = origin
    ux, uy = x_dir
    vx, vy = y_dir
    p1 = (x0, y0)
    p2 = (x0 + ux * dx, y0 + uy * dx)
    p3 = (p2[0] + vx * dy, p2[1] + vy * dy)
    p4 = (x0 + vx * dy, y0 + vy * dy)
    return [p1, p2, p3, p4]


def _hinge_adjacency(
    living_hinge_seams: list[SharedEdge],
) -> dict[str, list[tuple[str, SharedEdge]]]:
    adj: dict[str, list[tuple[str, SharedEdge]]] = {}
    for se in living_hinge_seams:
        adj.setdefault(se.panel_a, []).append((se.panel_b, se))
        adj.setdefault(se.panel_b, []).append((se.panel_a, se))
    return adj


def _connected_components(
    nodes: list[str],
    adj: dict[str, list[tuple[str, SharedEdge]]],
) -> list[list[str]]:
    seen: set[str] = set()
    comps: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        comp: list[str] = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt, _ in adj.get(cur, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        comps.append(comp)
    return comps


def _hinge_neighbor_transform(
    current: str,
    neighbor: str,
    se: SharedEdge,
    panel_map: dict[str, Panel2D],
    placed_outline: dict[str, list[tuple[float, float]]],
    placed_xform: dict[str, Affine2D],
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]], Affine2D] | None:
    cur_outline = placed_outline[current]
    cur_xform = placed_xform[current]
    cur_p2d = panel_map[current]
    nbr_p2d = panel_map[neighbor]

    se_cur_a = cur_p2d.project_3d(se.start_3d)
    se_cur_b = cur_p2d.project_3d(se.end_3d)
    se_nbr_a = nbr_p2d.project_3d(se.start_3d)
    se_nbr_b = nbr_p2d.project_3d(se.end_3d)

    se_svg_a = cur_xform.apply(*se_cur_a)
    se_svg_b = cur_xform.apply(*se_cur_b)
    se_svg_mid = ((se_svg_a[0] + se_svg_b[0]) / 2, (se_svg_a[1] + se_svg_b[1]) / 2)

    se_svg_dx = se_svg_b[0] - se_svg_a[0]
    se_svg_dy = se_svg_b[1] - se_svg_a[1]
    if math.hypot(se_svg_dx, se_svg_dy) < 1e-6:
        return None
    angle_svg = math.atan2(se_svg_dy, se_svg_dx)

    se_nbr_dx = se_nbr_b[0] - se_nbr_a[0]
    se_nbr_dy = se_nbr_b[1] - se_nbr_a[1]
    if math.hypot(se_nbr_dx, se_nbr_dy) < 1e-6:
        return None
    angle_nbr = math.atan2(se_nbr_dy, se_nbr_dx)

    out_n = _outward_normal_2d(cur_outline, se_svg_a, se_svg_b)
    align_rot = angle_svg - angle_nbr
    T_rot = Affine2D.from_rotation(align_rot)
    T_ref = Affine2D.from_reflection(angle_svg)
    T_rot_ref = T_ref.compose(T_rot)
    se_nbr_mid = ((se_nbr_a[0] + se_nbr_b[0]) / 2, (se_nbr_a[1] + se_nbr_b[1]) / 2)

    def _candidate(base: Affine2D) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]], Affine2D, tuple[float, float]]:
        xf_mid = base.apply(*se_nbr_mid)
        T_trans = Affine2D.from_translation(se_svg_mid[0] - xf_mid[0], se_svg_mid[1] - xf_mid[1])
        xform = T_trans.compose(base)
        final_outline = xform.apply_pts(nbr_p2d.outline)
        final_holes = [xform.apply_pts(h) for h in nbr_p2d.holes]

        others = [o for n, o in placed_outline.items() if n != current]
        overlap = _total_overlap_area(final_outline, others) if others else 0.0
        cx, cy = _poly_centroid_xy(final_outline)
        to_nbr = (cx - se_svg_mid[0], cy - se_svg_mid[1])
        outward = to_nbr[0] * out_n[0] + to_nbr[1] * out_n[1]
        # Prefer outward placement when overlaps are equivalent.
        score = (overlap, 0.0 if outward > 0 else 1.0)
        return final_outline, final_holes, xform, score

    cand_ref = _candidate(T_rot_ref)
    cand_rot = _candidate(T_rot)
    if cand_ref[3] <= cand_rot[3]:
        return cand_ref[0], cand_ref[1], cand_ref[2]
    return cand_rot[0], cand_rot[1], cand_rot[2]


def _hinge_slots_for_seam(
    se: SharedEdge,
    panel_map: dict[str, Panel2D],
    panel_xform: dict[str, Affine2D],
    piece_poly,
    thickness: float,
) -> list[list[tuple[float, float]]]:
    """Return living-hinge cut lines as 2-point polylines (not rectangular holes)."""
    if Polygon is None or LineString is None:
        return []
    if se.panel_a not in panel_map or se.panel_b not in panel_map:
        return []
    if se.panel_a not in panel_xform or se.panel_b not in panel_xform:
        return []

    pa = panel_map[se.panel_a]
    pb = panel_map[se.panel_b]
    xa = panel_xform[se.panel_a]
    xb = panel_xform[se.panel_b]

    a0 = xa.apply(*pa.project_3d(se.start_3d))
    a1 = xa.apply(*pa.project_3d(se.end_3d))
    dx = a1[0] - a0[0]
    dy = a1[1] - a0[1]
    seam_len = math.hypot(dx, dy)
    if seam_len < 8.0:
        return []

    t = (dx / seam_len, dy / seam_len)
    na = _outward_normal_2d(xa.apply_pts(pa.outline), a0, a1)
    cb = _poly_centroid_xy(xb.apply_pts(pb.outline))
    mid = ((a0[0] + a1[0]) / 2.0, (a0[1] + a1[1]) / 2.0)
    if na[0] * (cb[0] - mid[0]) + na[1] * (cb[1] - mid[1]) < 0:
        na = (-na[0], -na[1])

    # Fusion-style lattice parameters: staggered rows across a hinge band.
    band_half = max(3.2, thickness * 1.4)
    row_pitch = max(1.9, thickness * 0.6)
    slot_len = max(7.5, thickness * 2.6)
    link_gap = max(1.6, thickness * 0.5)

    y_start = -band_half
    y_end = band_half
    if y_end <= y_start:
        return []

    row_count = int((y_end - y_start) / row_pitch) + 1
    if row_count < 2:
        row_count = 2
    total_rows_span = (row_count - 1) * row_pitch
    y0 = -total_rows_span / 2.0

    period = slot_len + link_gap
    if period <= 1e-6:
        return []

    def _pattern_intervals(length: float, phase: float) -> list[tuple[float, float]]:
        if length <= 1e-6:
            return []
        low_k = int(math.floor((phase - slot_len) / period)) - 1
        high_k = int(math.ceil((length + phase) / period)) + 1
        intervals: list[tuple[float, float]] = []
        for k in range(low_k, high_k + 1):
            start = k * period - phase
            end = start + slot_len
            lo = max(0.0, start)
            hi = min(length, end)
            if hi - lo >= 0.8:
                intervals.append((lo, hi))
        if not intervals:
            return []
        intervals.sort()
        merged = [intervals[0]]
        for lo, hi in intervals[1:]:
            if lo <= merged[-1][1] + 1e-6:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    def _collect_segments(geom) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        if geom is None:
            return []
        gtype = getattr(geom, "geom_type", "")
        if gtype == "LineString":
            coords = list(geom.coords)
            if len(coords) >= 2:
                return [((float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1])))]
            return []
        if gtype == "MultiLineString" or gtype == "GeometryCollection":
            out: list[tuple[tuple[float, float], tuple[float, float]]] = []
            for g in geom.geoms:
                out.extend(_collect_segments(g))
            return out
        return []

    loops: list[list[tuple[float, float]]] = []
    min_x, min_y, max_x, max_y = piece_poly.bounds
    extend = math.hypot(max_x - min_x, max_y - min_y) + 10.0

    for row in range(row_count):
        y_off = y0 + row * row_pitch
        phase = 0.0 if row % 2 == 0 else period * 0.5

        p_start = (
            mid[0] - t[0] * extend + na[0] * y_off,
            mid[1] - t[1] * extend + na[1] * y_off,
        )
        p_end = (
            mid[0] + t[0] * extend + na[0] * y_off,
            mid[1] + t[1] * extend + na[1] * y_off,
        )

        row_line = LineString([p_start, p_end])
        inside = piece_poly.intersection(row_line)
        for s0, s1 in _collect_segments(inside):
            seg_vec = (s1[0] - s0[0], s1[1] - s0[1])
            seg_len = math.hypot(seg_vec[0], seg_vec[1])
            if seg_len < 0.8:
                continue

            ux, uy = (seg_vec[0] / seg_len, seg_vec[1] / seg_len)
            for lo, hi in _pattern_intervals(seg_len, phase):
                a = (s0[0] + ux * lo, s0[1] + uy * lo)
                b = (s0[0] + ux * hi, s0[1] + uy * hi)
                loops.append([a, b])

    return loops


def _build_piece_map(
    model: BinModel,
    panel_map: dict[str, Panel2D],
) -> dict[str, Piece2D]:
    names = list(panel_map.keys())
    seams = [
        se for se in model.living_hinge_seams
        if se.panel_a in panel_map and se.panel_b in panel_map
    ]
    if not seams or Polygon is None or unary_union is None or LineString is None:
        return {
            n: Piece2D(name=n, outline=list(panel_map[n].outline), holes=[list(h) for h in panel_map[n].holes])
            for n in names
        }

    hinge_adj = _hinge_adjacency(seams)

    pieces: dict[str, Piece2D] = {}
    for comp in _connected_components(names, hinge_adj):
        if len(comp) == 1:
            n = comp[0]
            pieces[n] = Piece2D(name=n, outline=list(panel_map[n].outline), holes=[list(h) for h in panel_map[n].holes])
            continue

        comp_set = set(comp)
        root = sorted(comp)[0]
        placed_outline: dict[str, list[tuple[float, float]]] = {root: list(panel_map[root].outline)}
        placed_holes: dict[str, list[list[tuple[float, float]]]] = {root: [list(h) for h in panel_map[root].holes]}
        placed_xform: dict[str, Affine2D] = {root: Affine2D.identity()}

        queue: deque[str] = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor, se in hinge_adj.get(current, []):
                if neighbor not in comp_set or neighbor in placed_outline:
                    continue
                placed = _hinge_neighbor_transform(
                    current=current,
                    neighbor=neighbor,
                    se=se,
                    panel_map=panel_map,
                    placed_outline=placed_outline,
                    placed_xform=placed_xform,
                )
                if placed is None:
                    continue
                out, holes, xform = placed
                placed_outline[neighbor] = out
                placed_holes[neighbor] = holes
                placed_xform[neighbor] = xform
                queue.append(neighbor)

        # Fallback: keep any unresolved members separate in this piece frame.
        for n in comp:
            if n in placed_outline:
                continue
            placed_outline[n] = list(panel_map[n].outline)
            placed_holes[n] = [list(h) for h in panel_map[n].holes]
            placed_xform[n] = Affine2D.identity()

        polys = []
        for n in comp:
            out = placed_outline[n]
            holes = placed_holes.get(n, [])
            try:
                poly = Polygon(out, holes)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty and poly.area > 1e-6:
                    polys.append(poly)
            except Exception:
                continue

        seam_bridges = []
        for a in comp:
            for b, se in hinge_adj.get(a, []):
                if b not in comp_set or a > b:
                    continue
                pa = panel_map[se.panel_a]
                xa = placed_xform[se.panel_a]
                p0 = xa.apply(*pa.project_3d(se.start_3d))
                p1 = xa.apply(*pa.project_3d(se.end_3d))
                try:
                    seam_bridges.append(LineString([p0, p1]).buffer(max(0.08, model.thickness * 0.03), cap_style=2))
                except Exception:
                    continue

        merged = unary_union(polys + seam_bridges)
        if hasattr(merged, "buffer"):
            merged = merged.buffer(0)
        if merged.geom_type == "MultiPolygon":
            try:
                merged = max(merged.geoms, key=lambda g: g.area)
            except Exception:
                pass
        if merged.geom_type != "Polygon":
            # Conservative fallback: keep root panel as piece if merge fails.
            n = root
            pieces[n] = Piece2D(name=n, outline=list(panel_map[n].outline), holes=[list(h) for h in panel_map[n].holes])
            continue

        piece_outline = _ring_to_loop(merged.exterior)
        piece_holes = [_ring_to_loop(r) for r in merged.interiors]
        piece_holes = [h for h in piece_holes if len(h) >= 3]

        # Add hinge slit lattice holes for all seams inside this component.
        for se in seams:
            if se.panel_a not in comp_set or se.panel_b not in comp_set:
                continue
            piece_holes.extend(
                _hinge_slots_for_seam(
                    se,
                    panel_map,
                    placed_xform,
                    merged,
                    model.thickness,
                )
            )

        piece_name = "+".join(sorted(comp))
        pieces[piece_name] = Piece2D(
            name=piece_name,
            outline=piece_outline,
            holes=piece_holes,
        )

    return pieces


def _build_local_axes(
    normal: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    nx, ny, nz = normal
    ref = (1.0, 0.0, 0.0) if abs(nx) < 0.9 else (0.0, 1.0, 0.0)
    u = _vec_cross(ref, normal)
    u_len = _vec_len(u)
    u = (u[0] / u_len, u[1] / u_len, u[2] / u_len)
    v = _vec_cross(normal, u)
    v_len = _vec_len(v)
    v = (v[0] / v_len, v[1] / v_len, v[2] / v_len)
    return u, v


def _wire_vertices_3d(wire: cq.Wire) -> list[tuple[float, float, float]]:
    explorer = BRepTools_WireExplorer(wire.wrapped)
    verts_3d: list[tuple[float, float, float]] = []
    while explorer.More():
        vert = explorer.CurrentVertex()
        pnt = BRep_Tool.Pnt_s(vert)
        verts_3d.append((pnt.X(), pnt.Y(), pnt.Z()))
        explorer.Next()
    return verts_3d


def _project_panel(
    solid: cq.Solid,
    outer_normal: tuple[float, float, float],
    name: str,
    frame: Panel2D | None = None,
) -> Panel2D | None:
    wp = cq.Workplane().add(solid)
    faces = wp.faces().vals()

    best_face = None
    best_area = 0.0
    for f in faces:
        try:
            n = f.normalAt()
        except Exception:
            continue
        dot = n.x * outer_normal[0] + n.y * outer_normal[1] + n.z * outer_normal[2]
        if dot > 0.99:
            a = f.Area()
            if a > best_area:
                best_area = a
                best_face = f
    if best_face is None:
        return None

    outer_wire = best_face.outerWire()
    outer_verts_3d = _wire_vertices_3d(outer_wire)
    if not outer_verts_3d:
        return None

    hole_wires = list(best_face.innerWires())
    hole_verts_3d = [_wire_vertices_3d(w) for w in hole_wires]

    if frame is not None:
        # Project into an existing local frame so overlays align exactly.
        u = frame.u_axis
        v = frame.v_axis
        min_x = frame.offset_x
        min_y = frame.offset_y
        pts_2d = [(_vec_dot(pt, u) - min_x, _vec_dot(pt, v) - min_y) for pt in outer_verts_3d]
        pts_2d = _collapse_short_segments(pts_2d)
        holes_2d = [
            _collapse_short_segments([(_vec_dot(pt, u) - min_x, _vec_dot(pt, v) - min_y) for pt in verts])
            for verts in hole_verts_3d
            if len(verts) >= 3
        ]
        holes_2d = _filter_boundary_touching_holes(pts_2d, holes_2d)
        return Panel2D(
            name=name, outline=pts_2d, holes=holes_2d,
            u_axis=u, v_axis=v,
            offset_x=min_x, offset_y=min_y,
        )

    u, v = _build_local_axes(outer_normal)
    raw_outline = [(_vec_dot(pt, u), _vec_dot(pt, v)) for pt in outer_verts_3d]
    raw_holes = [
        [(_vec_dot(pt, u), _vec_dot(pt, v)) for pt in verts]
        for verts in hole_verts_3d
        if len(verts) >= 3
    ]

    rot = _min_bbox_angle(raw_outline)
    outline_rot = _rotate_pts(raw_outline, -rot)
    holes_rot = [_rotate_pts(h, -rot) for h in raw_holes]

    min_x = min(p[0] for p in outline_rot)
    min_y = min(p[1] for p in outline_rot)
    pts_2d = [(p[0] - min_x, p[1] - min_y) for p in outline_rot]
    pts_2d = _collapse_short_segments(pts_2d)
    holes_2d = [
        _collapse_short_segments([(p[0] - min_x, p[1] - min_y) for p in hole])
        for hole in holes_rot
        if len(hole) >= 3
    ]
    holes_2d = _filter_boundary_touching_holes(pts_2d, holes_2d)

    # Rotated 3D axes:
    #   final_x = dot(P, u)*cos(-rot) - dot(P, v)*sin(-rot) - min_x
    #   final_y = dot(P, u)*sin(-rot) + dot(P, v)*cos(-rot) - min_y
    cos_r = math.cos(-rot)
    sin_r = math.sin(-rot)
    u_rot = tuple(cos_r * u[k] - sin_r * v[k] for k in range(3))
    v_rot = tuple(sin_r * u[k] + cos_r * v[k] for k in range(3))

    return Panel2D(
        name=name, outline=pts_2d, holes=holes_2d,
        u_axis=u_rot, v_axis=v_rot,
        offset_x=min_x, offset_y=min_y,
    )


# ---------------------------------------------------------------------------
# Unfolded layout via BFS
# ---------------------------------------------------------------------------

def _build_adjacency(
    shared_edges: list[SharedEdge],
) -> dict[str, list[tuple[str, SharedEdge]]]:
    adj: dict[str, list[tuple[str, SharedEdge]]] = {}
    for se in shared_edges:
        adj.setdefault(se.panel_a, []).append((se.panel_b, se))
        adj.setdefault(se.panel_b, []).append((se.panel_a, se))
    return adj


def _outward_normal_2d(
    outline: list[tuple[float, float]],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """Unit normal of segment a->b pointing away from polygon interior."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (0.0, 1.0)
    n1 = (-dy / length, dx / length)
    n2 = (dy / length, -dx / length)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    eps = max(0.25, min(2.0, length * 0.05))

    p1 = (mid[0] + n1[0] * eps, mid[1] + n1[1] * eps)
    p2 = (mid[0] + n2[0] * eps, mid[1] + n2[1] * eps)
    inside1 = _point_in_polygon(p1, outline)
    inside2 = _point_in_polygon(p2, outline)

    if inside1 and not inside2:
        return n2
    if inside2 and not inside1:
        return n1

    # Fallback for ambiguous cases.
    cx = sum(p[0] for p in outline) / len(outline)
    cy = sum(p[1] for p in outline) / len(outline)
    to_c = (cx - mid[0], cy - mid[1])
    if n1[0] * to_c[0] + n1[1] * to_c[1] < 0:
        return n1
    return n2


def _compute_unfolded_layout(
    model: BinModel,
    panel_map: dict[str, Panel2D],
    gap: float = 4.0,
) -> list[tuple[str, list[tuple[float, float]], list[list[tuple[float, float]]], Affine2D]]:
    """BFS from the bottom panel, placing neighbours next to shared edges.

    Each neighbour is reflected across the shared edge and shifted outward
    by *gap* mm, as if the 3D box were unfolded flat.
    """
    adj = _build_adjacency(model.shared_edges)
    root = "bottom" if "bottom" in panel_map else next(iter(panel_map))

    # For each placed panel: final SVG outline + Affine2D that maps local -> SVG
    placed_outline: dict[str, list[tuple[float, float]]] = {}
    placed_holes: dict[str, list[list[tuple[float, float]]]] = {}
    placed_xform: dict[str, Affine2D] = {}

    # Root at the origin
    placed_outline[root] = list(panel_map[root].outline)
    placed_holes[root] = [list(h) for h in panel_map[root].holes]
    placed_xform[root] = Affine2D.identity()

    queue: deque[str] = deque([root])
    visited: set[str] = {root}

    while queue:
        current = queue.popleft()
        if current not in adj:
            continue

        cur_outline = placed_outline[current]
        cur_xform = placed_xform[current]
        cur_p2d = panel_map[current]

        for neighbor, se in adj[current]:
            if neighbor in visited or neighbor not in panel_map:
                continue

            nbr_p2d = panel_map[neighbor]

            # Project shared edge into both panels' local 2D
            se_cur_a = cur_p2d.project_3d(se.start_3d)
            se_cur_b = cur_p2d.project_3d(se.end_3d)
            se_nbr_a = nbr_p2d.project_3d(se.start_3d)
            se_nbr_b = nbr_p2d.project_3d(se.end_3d)

            # Transform current panel's SE to SVG coords
            se_svg_a = cur_xform.apply(*se_cur_a)
            se_svg_b = cur_xform.apply(*se_cur_b)
            se_svg_mid = ((se_svg_a[0] + se_svg_b[0]) / 2,
                          (se_svg_a[1] + se_svg_b[1]) / 2)

            # SE direction angles
            se_svg_dx = se_svg_b[0] - se_svg_a[0]
            se_svg_dy = se_svg_b[1] - se_svg_a[1]
            if math.hypot(se_svg_dx, se_svg_dy) < 1e-6:
                continue
            angle_svg = math.atan2(se_svg_dy, se_svg_dx)

            se_nbr_dx = se_nbr_b[0] - se_nbr_a[0]
            se_nbr_dy = se_nbr_b[1] - se_nbr_a[1]
            if math.hypot(se_nbr_dx, se_nbr_dy) < 1e-6:
                continue
            angle_nbr = math.atan2(se_nbr_dy, se_nbr_dx)

            out_n = _outward_normal_2d(cur_outline, se_svg_a, se_svg_b)

            # --- Build the neighbour's transform ---
            # Goal: rotate + reflect the neighbour so its SE aligns with the
            # SVG SE, and its body extends in the outward direction.
            #
            # 1. Rotate to align SE directions (make them parallel).
            # 2. Reflect across the SE direction so the body flips outward.
            # 3. Translate so SE midpoints are gap apart along outward normal.
            #
            # The composed transform (applied right-to-left):
            #   T_translate * T_reflect * T_rotate
            # maps neighbour-local -> SVG.

            align_rot = angle_svg - angle_nbr
            T_rot = Affine2D.from_rotation(align_rot)
            T_ref = Affine2D.from_reflection(angle_svg)
            T_rot_ref = T_ref.compose(T_rot)

            # Where does the SE midpoint end up after rot+ref?
            se_nbr_mid = ((se_nbr_a[0] + se_nbr_b[0]) / 2,
                          (se_nbr_a[1] + se_nbr_b[1]) / 2)
            xf_mid = T_rot_ref.apply(*se_nbr_mid)

            target = (se_svg_mid[0] + out_n[0] * gap,
                      se_svg_mid[1] + out_n[1] * gap)
            T_trans = Affine2D.from_translation(target[0] - xf_mid[0],
                                                target[1] - xf_mid[1])
            nbr_xform = T_trans.compose(T_rot_ref)

            final_outline = nbr_xform.apply_pts(nbr_p2d.outline)
            final_holes = [nbr_xform.apply_pts(h) for h in nbr_p2d.holes]

            # Verify neighbour body extends outward
            nbr_cx = sum(p[0] for p in final_outline) / len(final_outline)
            nbr_cy = sum(p[1] for p in final_outline) / len(final_outline)
            to_nbr = (nbr_cx - se_svg_mid[0], nbr_cy - se_svg_mid[1])
            body_outward = to_nbr[0] * out_n[0] + to_nbr[1] * out_n[1]

            if body_outward < 0:
                # Body ended up on wrong side -- use rotation only (no reflection)
                xf_mid2 = T_rot.apply(*se_nbr_mid)
                T_trans2 = Affine2D.from_translation(target[0] - xf_mid2[0],
                                                     target[1] - xf_mid2[1])
                nbr_xform = T_trans2.compose(T_rot)
                final_outline = nbr_xform.apply_pts(nbr_p2d.outline)
                final_holes = [nbr_xform.apply_pts(h) for h in nbr_p2d.holes]

            # If this panel still overlaps existing placed panels, push it farther
            # along the outward normal until clear.
            others = list(placed_outline.values())
            final_outline, pushed = _push_out_until_clear(
                final_outline,
                out_n,
                others,
                step=max(gap, 2.0),
                min_clearance=max(0.0, gap),
            )
            if pushed > 0:
                T_push = Affine2D.from_translation(out_n[0] * pushed, out_n[1] * pushed)
                nbr_xform = T_push.compose(nbr_xform)
                final_holes = [
                    _translate_pts(h, out_n[0] * pushed, out_n[1] * pushed)
                    for h in final_holes
                ]

            placed_outline[neighbor] = final_outline
            placed_holes[neighbor] = final_holes
            placed_xform[neighbor] = nbr_xform
            visited.add(neighbor)
            queue.append(neighbor)

    # Place unreachable panels below the rest
    for name in panel_map:
        if name not in placed_outline:
            p = panel_map[name]
            all_y = [pt[1] for pts in placed_outline.values() for pt in pts]
            y_off = max(all_y) + 20.0 if all_y else 0.0
            placed_outline[name] = _translate_pts(p.outline, 0, y_off)
            placed_holes[name] = [_translate_pts(h, 0, y_off) for h in p.holes]
            placed_xform[name] = Affine2D.from_translation(0, y_off)

    # Shift so all coordinates are positive
    all_x = [pt[0] for pts in placed_outline.values() for pt in pts]
    all_y = [pt[1] for pts in placed_outline.values() for pt in pts]
    if all_x and all_y:
        sx = -min(all_x) + 5.0
        sy = -min(all_y) + 5.0
        for name in placed_outline:
            placed_outline[name] = _translate_pts(placed_outline[name], sx, sy)
            placed_holes[name] = [_translate_pts(h, sx, sy) for h in placed_holes.get(name, [])]
            placed_xform[name] = Affine2D.from_translation(sx, sy).compose(placed_xform[name])

    return [
        (name, placed_outline[name], placed_holes.get(name, []), placed_xform[name])
        for name in placed_outline
    ]


# ---------------------------------------------------------------------------
# Packed layout (sheet nesting)
# ---------------------------------------------------------------------------

def _poly_area_abs(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    acc = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def _bounds_xy(pts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0.0, max(ax0 - bx1, bx0 - ax1))
    dy = max(0.0, max(ay0 - by1, by0 - ay1))
    return math.hypot(dx, dy)


def _bbox_intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    eps: float = 1e-6,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 <= bx0 + eps or bx1 <= ax0 + eps:
        return False
    if ay1 <= by0 + eps or by1 <= ay0 + eps:
        return False
    return True


def _outline_poly(pts: list[tuple[float, float]]):
    if Polygon is None or len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-6:
            return None
        return poly
    except Exception:
        return None


def _candidate_points_for_sheet(
    existing_bounds: list[tuple[float, float, float, float]],
    margin: float,
    part_gap: float,
) -> list[tuple[float, float]]:
    candidates: set[tuple[float, float]] = {(margin, margin)}
    xs: set[float] = {margin}
    ys: set[float] = {margin}
    for min_x, min_y, max_x, max_y in existing_bounds:
        candidates.add((max_x + part_gap, min_y))
        candidates.add((min_x, max_y + part_gap))
        candidates.add((max_x + part_gap, max_y + part_gap))
        xs.add(min_x)
        xs.add(max_x + part_gap)
        ys.add(min_y)
        ys.add(max_y + part_gap)

    for x in xs:
        for y in ys:
            candidates.add((x, y))

    return sorted(candidates, key=lambda p: (p[1], p[0]))


def _make_oriented_geometry(
    panel: Panel2D | Piece2D,
    angle_rad: float,
) -> dict[str, object]:
    rot = Affine2D.from_rotation(angle_rad)
    out_rot = rot.apply_pts(panel.outline)
    holes_rot = [rot.apply_pts(h) for h in panel.holes]
    min_x, min_y, max_x, max_y = _bounds_xy(out_rot)
    out_norm = _translate_pts(out_rot, -min_x, -min_y)
    holes_norm = [_translate_pts(h, -min_x, -min_y) for h in holes_rot]
    return {
        "outline": out_norm,
        "holes": holes_norm,
        "width": max_x - min_x,
        "height": max_y - min_y,
        "xform": Affine2D.from_translation(-min_x, -min_y).compose(rot),
    }


def _best_required_sheet_size(
    dims: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Return the most compact (width, height, angle_deg) from oriented dims."""
    if not dims:
        return (0.0, 0.0, 0.0)
    return min(
        dims,
        key=lambda d: (
            d[0] * d[1],           # smallest bounding-box area
            max(d[0], d[1]),       # then smallest longest side
            d[0] + d[1],           # then smallest perimeter proxy
        ),
    )


def _sheet_origins(
    n_sheets: int,
    sheet_width: float,
    sheet_height: float,
    sheet_gap: float,
) -> list[tuple[float, float]]:
    if n_sheets <= 0:
        return []
    cols = max(1, math.ceil(math.sqrt(n_sheets)))
    origins: list[tuple[float, float]] = []
    for idx in range(n_sheets):
        col = idx % cols
        row = idx // cols
        origins.append((col * (sheet_width + sheet_gap), row * (sheet_height + sheet_gap)))
    return origins


def _compute_packed_layout(
    panel_map: dict[str, Panel2D | Piece2D],
    sheet_width: float,
    sheet_height: float,
    part_gap: float = 4.0,
    sheet_gap: float = 15.0,
    pack_rotations: int = 2,
) -> tuple[
    list[tuple[str, list[tuple[float, float]], list[list[tuple[float, float]]], Affine2D]],
    list[tuple[float, float, float, float]],
]:
    """Pack flat panels into one or more fixed-size sheets.

    Uses a bottom-left heuristic on candidate anchors and supports optional
    part rotation steps (default: 0/90 degrees).
    """
    if sheet_width <= 0 or sheet_height <= 0:
        raise ValueError("sheet_width and sheet_height must be > 0 for packed layout")

    if pack_rotations <= 1:
        angles = [0.0]
    elif pack_rotations == 2:
        angles = [0.0, math.pi / 2.0]
    else:
        step = 2.0 * math.pi / float(pack_rotations)
        angles = [i * step for i in range(pack_rotations)]

    oriented_by_name: dict[str, list[dict[str, object]]] = {}
    for name, panel in panel_map.items():
        variants: list[dict[str, object]] = []
        dims_all: list[tuple[float, float, float]] = []
        for angle in angles:
            g = _make_oriented_geometry(panel, angle)
            width = float(g["width"])
            height = float(g["height"])
            angle_deg = (math.degrees(angle) % 360.0)
            dims_all.append((width, height, angle_deg))
            if width <= sheet_width + 1e-6 and height <= sheet_height + 1e-6:
                variants.append(g)
        if not variants:
            req_w, req_h, req_angle = _best_required_sheet_size(dims_all)
            raise ValueError(
                f"Panel '{name}' does not fit into the requested sheet size "
                f"({sheet_width} x {sheet_height} mm). "
                f"Minimum needed (with current rotations) is about "
                f"{req_w:.1f} x {req_h:.1f} mm at {req_angle:.1f}°."
            )
        oriented_by_name[name] = variants

    order = sorted(
        panel_map.keys(),
        key=lambda n: _poly_area_abs(panel_map[n].outline),
        reverse=True,
    )

    edge_margin = 0.0

    def _best_on_sheet(
        sheet_states: list[dict[str, list]],
        sheet_idx: int,
        name: str,
    ) -> dict[str, object] | None:
        state = sheet_states[sheet_idx]
        candidates = _candidate_points_for_sheet(state["bounds"], edge_margin, part_gap)
        best: dict[str, object] | None = None
        best_score: tuple[float, float, float, float] | None = None
        occ_x = max((b[2] for b in state["bounds"]), default=0.0)
        occ_y = max((b[3] for b in state["bounds"]), default=0.0)

        def _try_position(x: float, y: float, orient: dict[str, object]) -> None:
            nonlocal best, best_score
            out = orient["outline"]  # type: ignore[assignment]
            holes = orient["holes"]  # type: ignore[assignment]
            w = float(orient["width"])
            h = float(orient["height"])
            if x < edge_margin - 1e-6 or y < edge_margin - 1e-6:
                return
            if x + w > sheet_width - edge_margin + 1e-6:
                return
            if y + h > sheet_height - edge_margin + 1e-6:
                return

            bounds = (x, y, x + w, y + h)
            out_xy = _translate_pts(out, x, y)  # type: ignore[arg-type]
            cand_poly = _outline_poly(out_xy)
            if Polygon is not None and cand_poly is None:
                return

            for idx_existing, b in enumerate(state["bounds"]):
                if not _bbox_intersects(bounds, b):
                    if part_gap <= 0 or _bbox_distance(bounds, b) >= part_gap - 1e-6:
                        continue
                if Polygon is None:
                    return
                ex_poly = state["polys"][idx_existing]
                if ex_poly is None:
                    continue
                if cand_poly.intersects(ex_poly):  # type: ignore[union-attr]
                    return
                if part_gap > 0 and cand_poly.distance(ex_poly) < part_gap - 1e-6:  # type: ignore[union-attr]
                    return

            holes_xy = [_translate_pts(hole, x, y) for hole in holes]  # type: ignore[arg-type]
            xform = Affine2D.from_translation(x, y).compose(orient["xform"])  # type: ignore[arg-type]
            new_occ_x = max(occ_x, x + w)
            new_occ_y = max(occ_y, y + h)
            score = (new_occ_x * new_occ_y, new_occ_y, new_occ_x, y)
            if best_score is None or score < best_score:
                best_score = score
                best = {
                    "name": name,
                    "sheet_index": sheet_idx,
                    "outline": out_xy,
                    "holes": holes_xy,
                    "xform": xform,
                    "bounds": bounds,
                    "poly": cand_poly,
                }

        # 1) Fast anchor candidates
        for x, y in candidates:
            for orient in oriented_by_name[name]:
                _try_position(x, y, orient)

        # 2) Dense scan fallback for irregular gaps (SheetPack-inspired)
        if best is None:
            step_sizes = [2.0, 1.0] if max(sheet_width, sheet_height) <= 1200 else [4.0, 2.0]
            for step in step_sizes:
                if best is not None:
                    break
                y = 0.0
                while y <= sheet_height + 1e-6:
                    x = 0.0
                    while x <= sheet_width + 1e-6:
                        for orient in oriented_by_name[name]:
                            _try_position(x, y, orient)
                        x += step
                    y += step

        return best

    def _pack_with_order(order_names: list[str]) -> tuple[list[dict[str, object]], list[dict[str, list]]]:
        sheet_states: list[dict[str, list]] = []
        placements_local: list[dict[str, object]] = []

        for name in order_names:
            placed = None
            for sheet_idx in range(len(sheet_states)):
                cand = _best_on_sheet(sheet_states, sheet_idx, name)
                if cand is not None:
                    placed = cand
                    break  # Prefer earliest existing sheet to minimize sheet count.

            if placed is None:
                sheet_states.append({"bounds": [], "polys": []})
                placed = _best_on_sheet(sheet_states, len(sheet_states) - 1, name)
                if placed is None:
                    raise ValueError(
                        f"Failed to place panel '{name}' on a new sheet "
                        f"({sheet_width} x {sheet_height} mm)"
                    )

            sheet_idx = int(placed["sheet_index"])
            sheet_states[sheet_idx]["bounds"].append(placed["bounds"])
            sheet_states[sheet_idx]["polys"].append(placed["poly"])
            placements_local.append(placed)

        return placements_local, sheet_states

    best_placements, best_sheets = _pack_with_order(order)

    # Try additional deterministic orders (inspiration from SheetPack multi-attempt search).
    if len(order) <= 6 and len(best_sheets) > 1:
        fixed = [order[0]]
        rest = order[1:]
        for perm in permutations(rest):
            trial_order = fixed + list(perm)
            trial_placements, trial_sheets = _pack_with_order(trial_order)
            if len(trial_sheets) < len(best_sheets):
                best_placements, best_sheets = trial_placements, trial_sheets
                if len(best_sheets) == 1:
                    break

    placements_local = best_placements
    sheet_states = best_sheets

    origins = _sheet_origins(len(sheet_states), sheet_width, sheet_height, sheet_gap)

    placed_global: list[
        tuple[str, list[tuple[float, float]], list[list[tuple[float, float]]], Affine2D]
    ] = []
    for placed in placements_local:
        sheet_idx = int(placed["sheet_index"])
        ox, oy = origins[sheet_idx]
        out_g = _translate_pts(placed["outline"], ox, oy)  # type: ignore[arg-type]
        holes_g = [_translate_pts(h, ox, oy) for h in placed["holes"]]  # type: ignore[arg-type]
        xform_g = Affine2D.from_translation(ox, oy).compose(placed["xform"])  # type: ignore[arg-type]
        placed_global.append((str(placed["name"]), out_g, holes_g, xform_g))

    sheet_rects = [
        (ox, oy, sheet_width, sheet_height)
        for ox, oy in origins
    ]
    return placed_global, sheet_rects


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def export_svg(
    model: BinModel,
    output_path: str,
    spacing: float = 5.0,
    reference_model: BinModel | None = None,
    layout: str = "unfolded",
    sheet_width: float | None = None,
    sheet_height: float | None = None,
    part_gap: float = 4.0,
    sheet_gap: float = 15.0,
    pack_rotations: int = 2,
) -> None:
    """Export all panels as flat 2D outlines into an SVG file.

    Supported layouts:
    - ``unfolded``: adjacency-preserving unfold with fixed seam gaps
    - ``packed``: sheet nesting into one or more fixed-size sheets
    """
    if svgwrite is None:
        raise ImportError("svgwrite is required for SVG export (pip install svgwrite)")

    panel_map: dict[str, Panel2D] = {}
    for name, panel in model.panels.items():
        p2d = _project_panel(panel.solid, panel.outer_normal, name)
        if p2d is not None:
            panel_map[name] = p2d

    if not panel_map:
        raise ValueError("No outlines could be projected")

    sheet_rects: list[tuple[float, float, float, float]] = []
    if layout == "packed":
        if sheet_width is None or sheet_height is None:
            raise ValueError("packed layout requires sheet_width and sheet_height")
        packed_map: dict[str, Panel2D | Piece2D] = _build_piece_map(model, panel_map)
        placed, sheet_rects = _compute_packed_layout(
            packed_map,
            sheet_width=sheet_width,
            sheet_height=sheet_height,
            part_gap=max(0.0, part_gap),
            sheet_gap=max(0.0, sheet_gap),
            pack_rotations=max(1, int(pack_rotations)),
        )
    else:
        placed = _compute_unfolded_layout(model, panel_map, gap=4.0)

    if not placed:
        raise ValueError("No panels could be placed")

    all_x = [p[0] for _, pts, _, _ in placed for p in pts]
    all_y = [p[1] for _, pts, _, _ in placed for p in pts]
    for x, y, w, h in sheet_rects:
        all_x.extend([x, x + w])
        all_y.extend([y, y + h])

    min_x = min(all_x)
    min_y = min(all_y)
    shift_x = -min_x + spacing if min_x < 0 else 0.0
    shift_y = -min_y + spacing if min_y < 0 else 0.0
    if shift_x > 0 or shift_y > 0:
        placed = [
            (
                name,
                _translate_pts(pts, shift_x, shift_y),
                [_translate_pts(hole, shift_x, shift_y) for hole in holes],
                Affine2D.from_translation(shift_x, shift_y).compose(xform),
            )
            for name, pts, holes, xform in placed
        ]
        sheet_rects = [
            (x + shift_x, y + shift_y, w, h)
            for x, y, w, h in sheet_rects
        ]
        all_x = [p[0] for _, pts, _, _ in placed for p in pts]
        all_y = [p[1] for _, pts, _, _ in placed for p in pts]
        for x, y, w, h in sheet_rects:
            all_x.extend([x, x + w])
            all_y.extend([y, y + h])

    total_w = max(all_x) + spacing
    total_h = max(all_y) + spacing

    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{total_w}mm", f"{total_h}mm"),
        viewBox=f"0 0 {total_w} {total_h}",
    )

    # Explicit white background improves visibility in dark-theme viewers.
    dwg.add(dwg.rect(
        insert=(0, 0),
        size=(total_w, total_h),
        fill="#FFFFFF",
        stroke="none",
    ))

    if sheet_rects:
        for idx, (x, y, w, h) in enumerate(sheet_rects):
            dwg.add(dwg.rect(
                insert=(x, y),
                size=(w, h),
                fill="none",
                stroke="#B9B9B9",
                stroke_width="0.15",
                stroke_dasharray="2.0,1.5",
            ))
            dwg.add(dwg.text(
                f"sheet {idx + 1}",
                insert=(x + 2.5, y + 2.5),
                text_anchor="start",
                dominant_baseline="hanging",
                font_size="3.2",
                fill="#8A8A8A",
            ))

    for name, pts, holes, xform in placed:
        if not pts:
            continue
        d_parts = [f"M {pts[0][0]:.4f},{pts[0][1]:.4f}"]
        for p in pts[1:]:
            d_parts.append(f"L {p[0]:.4f},{p[1]:.4f}")
        d_parts.append("Z")

        dwg.add(dwg.path(
            d=" ".join(d_parts),
            stroke="#000000",
            stroke_width="0.1",
            fill="none",
        ))

        if reference_model is not None and name in reference_model.panels and name in panel_map:
            ref_panel = reference_model.panels[name]
            ref_p2d = _project_panel(
                ref_panel.solid,
                ref_panel.outer_normal,
                name,
                frame=panel_map[name],
            )
            if ref_p2d is not None and ref_p2d.outline:
                ref_outline = xform.apply_pts(ref_p2d.outline)
                r_parts = [f"M {ref_outline[0][0]:.4f},{ref_outline[0][1]:.4f}"]
                for p in ref_outline[1:]:
                    r_parts.append(f"L {p[0]:.4f},{p[1]:.4f}")
                r_parts.append("Z")
                dwg.add(dwg.path(
                    d=" ".join(r_parts),
                    stroke="#1BAA5C",
                    stroke_width="0.12",
                    stroke_dasharray="2.0,1.6",
                    fill="none",
                    opacity="0.9",
                ))

        for hole in holes:
            if len(hole) == 2:
                dwg.add(dwg.path(
                    d=f"M {hole[0][0]:.4f},{hole[0][1]:.4f} L {hole[1][0]:.4f},{hole[1][1]:.4f}",
                    stroke="#000000",
                    stroke_width="0.1",
                    fill="none",
                ))
                continue
            if len(hole) < 3:
                continue
            h_parts = [f"M {hole[0][0]:.4f},{hole[0][1]:.4f}"]
            for p in hole[1:]:
                h_parts.append(f"L {p[0]:.4f},{p[1]:.4f}")
            h_parts.append("Z")
            dwg.add(dwg.path(
                d=" ".join(h_parts),
                stroke="#000000",
                stroke_width="0.1",
                fill="none",
            ))

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        dwg.add(dwg.text(
            name, insert=(cx, cy),
            text_anchor="middle", dominant_baseline="central",
            font_size="4", fill="#222222",
        ))

    dwg.save()
