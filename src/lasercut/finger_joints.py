"""From-scratch finger joint generation.

Design goals for this reset:
- deterministic owner/mate seam contract
- one interval source of truth per seam (owner side)
- mapped mate slots derived from owner intervals only
- no legacy plateau/keepout heuristics
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .face_classifier import SharedEdge
from .projector import Projection2D, _dot, _normalize, _project_point, _sub
from .step_loader import PlanarFace

# Safe-zone defaults kept for CLI compatibility.
DEFAULT_NOTCH_BUFFER = 2.0
DEFAULT_PLATEAU_INSET = 3.0
DEFAULT_MIN_PLATEAU_LENGTH = 12.0

# Fusion-style option names kept for CLI compatibility.
FUSION_DYNAMIC_EQUAL = "equal"
FUSION_DYNAMIC_FIXED_NOTCH = "fixed_notch"
FUSION_DYNAMIC_FIXED_FINGER = "fixed_finger"
FUSION_DEFAULT_EDGE_MARGIN = 2.0

FUSION_PLACEMENT_FINGERS_OUTSIDE = "fingers_outside"
FUSION_PLACEMENT_NOTCHES_OUTSIDE = "notches_outside"
FUSION_PLACEMENT_SAME_START_FINGER = "same_start_finger"
FUSION_PLACEMENT_SAME_START_NOTCH = "same_start_notch"

TAB_DIRECTION_OUTWARD = "outward"
TAB_DIRECTION_INWARD = "inward"


@dataclass
class FusionJointParams:
    """Sizing/placement settings.

    The reset implementation honors these fields but uses a simpler,
    deterministic segmentation strategy.
    """

    placement_type: str = FUSION_PLACEMENT_FINGERS_OUTSIDE
    dynamic_size_type: str = FUSION_DYNAMIC_EQUAL
    is_number_of_fingers_fixed: bool = False
    fixed_num_fingers: int = 3
    fixed_finger_size: float = 20.0
    fixed_notch_size: float = 20.0
    min_finger_size: float = 20.0
    min_notch_size: float = 20.0
    gap: float = 0.0


def _dist_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _points_close_3d(a: tuple[float, float, float], b: tuple[float, float, float], tol: float = 0.5) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol and abs(a[2] - b[2]) <= tol


def _polygon_to_shapely(
    outer: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
) -> Polygon:
    if len(outer) < 3:
        return Polygon()
    return Polygon(outer, [h for h in (holes or []) if len(h) >= 3])


def _shapely_to_vertices(geom) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    if geom.is_empty:
        return [], []

    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    elif isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return [], []
        geom = max(polys, key=lambda g: g.area)

    outer = list(geom.exterior.coords[:-1])
    inners = [list(r.coords[:-1]) for r in geom.interiors]
    return outer, inners


def _find_matching_edge_index(proj: Projection2D, shared_edge: SharedEdge) -> int | None:
    tol = 0.5
    for idx, edge_3d in enumerate(proj.edge_map_3d):
        for se in (shared_edge.edge_a, shared_edge.edge_b):
            if _points_close_3d(edge_3d.midpoint, se.midpoint, tol):
                return idx

            same = _points_close_3d(edge_3d.start, se.start, tol) and _points_close_3d(edge_3d.end, se.end, tol)
            rev = _points_close_3d(edge_3d.start, se.end, tol) and _points_close_3d(edge_3d.end, se.start, tol)
            if same or rev:
                return idx
    return None


def _edges_reversed(
    proj_a: Projection2D,
    edge_idx_a: int,
    proj_b: Projection2D,
    edge_idx_b: int,
    tol: float = 0.5,
) -> bool:
    ea = proj_a.edge_map_3d[edge_idx_a]
    eb = proj_b.edge_map_3d[edge_idx_b]

    fwd = _points_close_3d(ea.start, eb.start, tol) and _points_close_3d(ea.end, eb.end, tol)
    rev = _points_close_3d(ea.start, eb.end, tol) and _points_close_3d(ea.end, eb.start, tol)
    return rev and not fwd


def _find_edge_index_by_endpoints(
    proj: Projection2D,
    p1: tuple[float, float],
    p2: tuple[float, float],
    tol: float = 0.5,
) -> int | None:
    for idx, (a, b) in enumerate(proj.outer_edges_2d):
        fwd = _dist_2d(a, p1) <= tol and _dist_2d(b, p2) <= tol
        rev = _dist_2d(a, p2) <= tol and _dist_2d(b, p1) <= tol
        if fwd or rev:
            return idx
    return None


def _outward_direction(
    p1: tuple[float, float],
    p2: tuple[float, float],
    polygon: Polygon,
) -> tuple[float, float]:
    """Outward perpendicular from edge p1->p2 by centroid distance heuristic."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (0.0, 1.0)

    n1 = (-dy / length, dx / length)
    n2 = (dy / length, -dx / length)

    mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    c = polygon.centroid

    d1 = (mid[0] + n1[0] - c.x) ** 2 + (mid[1] + n1[1] - c.y) ** 2
    d2 = (mid[0] + n2[0] - c.x) ** 2 + (mid[1] + n2[1] - c.y) ** 2
    return n1 if d1 > d2 else n2


def _build_exclusion_zones(proj: Projection2D, buffer_dist: float) -> list[Polygon]:
    zones: list[Polygon] = []
    if buffer_dist <= 0:
        return zones

    for inner in proj.inner_polygons:
        if len(inner) < 3:
            continue
        p = Polygon(inner)
        if p.is_valid and not p.is_empty:
            zones.append(p.buffer(buffer_dist))
    return zones


def _find_bottom_edge_endpoints(
    wall: PlanarFace,
    bottom: PlanarFace,
    tol: float = 5.0,
):
    """Find endpoints of wall edge nearest bottom plane (kept for exporter anchor logic)."""
    bn = _normalize(bottom.normal)

    def plane_dist(pt: tuple[float, float, float]) -> float:
        return _dot(bn, _sub(pt, bottom.center))

    edge_dists = [(abs(plane_dist(e.midpoint)), e) for e in wall.outer_wire_edges]
    if not edge_dists:
        return None

    min_dist = min(d for d, _ in edge_dists)
    near_edges = [e for d, e in edge_dists if d < min_dist + tol]
    if not near_edges:
        return None

    pts = [e.start for e in near_edges] + [e.end for e in near_edges]
    max_d = -1.0
    best_pair = None
    for i, p1 in enumerate(pts):
        for j, p2 in enumerate(pts):
            if j <= i:
                continue
            d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
            if d > max_d:
                max_d = d
                best_pair = (p1, p2)

    return best_pair


def _merge_intervals(intervals: list[tuple[float, float]], eps: float = 1e-6) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged: list[tuple[float, float]] = []
    for s, w in sorted(intervals):
        if w <= eps:
            continue
        e = s + w
        if not merged:
            merged.append((s, w))
            continue
        ps, pw = merged[-1]
        pe = ps + pw
        if s <= pe + eps:
            merged[-1] = (ps, max(pe, e) - ps)
        else:
            merged.append((s, w))
    return merged


def _clip_interval(
    start: float,
    end: float,
    lo: float,
    hi: float,
    eps: float = 1e-6,
) -> tuple[float, float] | None:
    s = max(start, lo)
    e = min(end, hi)
    if e <= s + eps:
        return None
    return (s, e)


def _intervals_to_spans(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(s, s + w) for s, w in intervals if w > 1e-6]


def _spans_to_intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(s, e - s) for s, e in spans if e > s + 1e-6]


def _complement_on_range(
    intervals: list[tuple[float, float]],
    lo: float,
    hi: float,
    eps: float = 1e-6,
) -> list[tuple[float, float]]:
    if hi <= lo + eps:
        return []

    spans = _intervals_to_spans(_merge_intervals(intervals, eps))
    spans = [s for s in spans if s[1] > lo + eps and s[0] < hi - eps]
    spans = [(max(lo, s0), min(hi, s1)) for s0, s1 in spans if min(hi, s1) > max(lo, s0) + eps]

    if not spans:
        return [(lo, hi - lo)]

    out: list[tuple[float, float]] = []
    cursor = lo
    for s0, s1 in spans:
        if s0 > cursor + eps:
            out.append((cursor, s0 - cursor))
        cursor = max(cursor, s1)
    if hi > cursor + eps:
        out.append((cursor, hi - cursor))
    return out


def _placement_flags(placement: str) -> tuple[bool, bool]:
    """Return (start_is_finger, end_is_finger)."""
    if placement == FUSION_PLACEMENT_NOTCHES_OUTSIDE:
        return (False, False)
    if placement == FUSION_PLACEMENT_SAME_START_FINGER:
        return (True, False)
    if placement == FUSION_PLACEMENT_SAME_START_NOTCH:
        return (False, True)
    return (True, True)


def _choose_segments_count(
    active_len: float,
    params: FusionJointParams,
    start_is_finger: bool,
    end_is_finger: bool,
) -> int:
    eps = 1e-6
    if active_len <= eps:
        return 0

    min_seg = max(1.0, min(params.min_finger_size, params.min_notch_size))

    if params.is_number_of_fingers_fixed:
        n_fingers = max(1, int(params.fixed_num_fingers))
        if start_is_finger and end_is_finger:
            n = 2 * n_fingers - 1
        elif (not start_is_finger) and (not end_is_finger):
            n = 2 * n_fingers + 1
        else:
            n = 2 * n_fingers
    else:
        n = max(2, int(active_len / min_seg))

    if n < 2:
        return 0

    def predicted_end(segments_count: int) -> bool:
        return start_is_finger if segments_count % 2 == 1 else (not start_is_finger)

    while n >= 2 and predicted_end(n) != end_is_finger:
        n -= 1

    if n < 2:
        return 0

    seg_w = active_len / n
    if seg_w < 1.0:
        return 0
    return n


def _build_owner_intervals(
    edge_len: float,
    params: FusionJointParams,
    start_margin: float,
    end_margin: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float, float] | None:
    eps = 1e-6
    if edge_len <= eps:
        return None

    lo = max(0.0, start_margin)
    hi = edge_len - max(0.0, end_margin)
    if hi <= lo + 2.0:
        return None

    active_len = hi - lo
    start_is_finger, end_is_finger = _placement_flags(params.placement_type)
    n = _choose_segments_count(active_len, params, start_is_finger, end_is_finger)
    if n <= 0:
        return None

    seg_w = active_len / n

    fingers: list[tuple[float, float]] = []
    for i in range(n):
        is_finger = (i % 2 == 0) if start_is_finger else (i % 2 == 1)
        if not is_finger:
            continue
        s = lo + i * seg_w
        fingers.append((s, seg_w))

    # Mate cut intervals derive from owner fingers (+/- gap), still clipped to active range.
    gap = max(0.0, params.gap)
    slots: list[tuple[float, float]] = []
    for s, w in fingers:
        span = _clip_interval(s - gap, s + w + gap, lo, hi)
        if span is not None:
            slots.append((span[0], span[1] - span[0]))

    return _merge_intervals(fingers), _merge_intervals(slots), lo, hi


def _map_intervals_by_param(
    src_len: float,
    dst_len: float,
    intervals: list[tuple[float, float]],
    reverse: bool,
) -> list[tuple[float, float]]:
    eps = 1e-6
    if src_len <= eps or dst_len <= eps or not intervals:
        return []

    mapped: list[tuple[float, float]] = []
    for s, w in intervals:
        if w <= eps:
            continue
        e = s + w
        t0 = max(0.0, min(1.0, s / src_len))
        t1 = max(0.0, min(1.0, e / src_len))
        if t1 <= t0 + eps:
            continue

        if reverse:
            m0 = dst_len * (1.0 - t1)
            m1 = dst_len * (1.0 - t0)
        else:
            m0 = dst_len * t0
            m1 = dst_len * t1

        if m1 > m0 + eps:
            mapped.append((m0, m1 - m0))

    return _merge_intervals(mapped)


def _make_comb_from_intervals(
    p1: tuple[float, float],
    p2: tuple[float, float],
    depth: float,
    outward_dir: tuple[float, float],
    intervals: list[tuple[float, float]],
    exclusion_zones: list[Polygon] | None = None,
    overlap: float = 0.01,
) -> list[Polygon]:
    """Build rectangular comb features on edge p1->p2."""
    teeth: list[Polygon] = []
    edge_len = _dist_2d(p1, p2)
    if edge_len < 1e-6 or depth <= 1e-9:
        return teeth

    ux = (p2[0] - p1[0]) / edge_len
    uy = (p2[1] - p1[1]) / edge_len
    nx, ny = outward_dir

    exclusion_union = unary_union(exclusion_zones) if exclusion_zones else None

    for s, w in intervals:
        if w <= 1e-6:
            continue
        a = max(0.0, s)
        b = min(edge_len, s + w)
        if b <= a + 1e-6:
            continue

        q0 = (p1[0] + ux * a, p1[1] + uy * a)
        q1 = (p1[0] + ux * b, p1[1] + uy * b)

        rect = Polygon(
            [
                (q0[0] - nx * overlap, q0[1] - ny * overlap),
                (q1[0] - nx * overlap, q1[1] - ny * overlap),
                (q1[0] + nx * depth, q1[1] + ny * depth),
                (q0[0] + nx * depth, q0[1] + ny * depth),
            ]
        )
        if rect.is_empty or not rect.is_valid or rect.area <= 1e-9:
            continue

        geom = rect
        if exclusion_union is not None:
            geom = geom.difference(exclusion_union)
            if geom.is_empty:
                continue

        if isinstance(geom, Polygon):
            geoms = [geom]
        elif isinstance(geom, MultiPolygon):
            geoms = list(geom.geoms)
        elif isinstance(geom, GeometryCollection):
            geoms = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        else:
            geoms = []

        for g in geoms:
            if g.is_valid and not g.is_empty and g.area > 1e-9:
                teeth.append(g)

    return teeth


def _clip_intervals_to_terminal_margins(
    edge_len: float,
    intervals: list[tuple[float, float]],
    start_margin: float,
    end_margin: float,
) -> list[tuple[float, float]]:
    lo = max(0.0, start_margin)
    hi = edge_len - max(0.0, end_margin)
    clipped: list[tuple[float, float]] = []
    for s, w in intervals:
        span = _clip_interval(s, s + w, lo, hi)
        if span is not None:
            clipped.append((span[0], span[1] - span[0]))
    return _merge_intervals(clipped)


def _find_plateau_segments(
    p1: tuple[float, float],
    p2: tuple[float, float],
    polygon: Polygon,
    sample_interval: float = 0.5,
    plateau_inset: float = DEFAULT_PLATEAU_INSET,
) -> list[tuple[float, float]]:
    """Compatibility stub for old overlap diagnostics.

    The reset engine does not rely on plateau analysis, so this returns an
    empty list to force full-edge contract generation.
    """
    _ = (p1, p2, polygon, sample_interval, plateau_inset)
    return []


def _intersect_segment_lists(
    segs_a: list[tuple[float, float]],
    segs_b: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    i = 0
    j = 0
    out: list[tuple[float, float]] = []
    while i < len(segs_a) and j < len(segs_b):
        a0, a1 = segs_a[i]
        b0, b1 = segs_b[j]
        s = max(a0, b0)
        e = min(a1, b1)
        if e > s + 1e-6:
            out.append((s, e))
        if a1 < b1:
            i += 1
        else:
            j += 1
    return out


def _reverse_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return sorted((1.0 - e, 1.0 - s) for s, e in segments)


def apply_finger_joints_fusion(
    projections: dict[int, Projection2D],
    shared_edges: list[SharedEdge],
    bottom_id: int,
    thickness: float,
    kerf: float = 0.0,
    edge_margin: float = -1,
    notch_buffer: float = -1,
    plateau_inset: float = -1,
    min_plateau_length: float = -1,
    tab_direction: str = TAB_DIRECTION_INWARD,
    faces: list[PlanarFace] | None = None,
    fusion_params: FusionJointParams | None = None,
) -> tuple[dict[int, list[tuple[float, float]]], dict[int, list[list[tuple[float, float]]]]]:
    """Generate finger joints from scratch with owner-driven seam contracts."""
    _ = (plateau_inset, min_plateau_length)

    if fusion_params is None:
        fusion_params = FusionJointParams()

    if edge_margin < 0:
        edge_margin = FUSION_DEFAULT_EDGE_MARGIN
    if notch_buffer < 0:
        notch_buffer = DEFAULT_NOTCH_BUFFER
    if tab_direction not in (TAB_DIRECTION_OUTWARD, TAB_DIRECTION_INWARD):
        raise ValueError(
            f"Unsupported tab_direction={tab_direction!r}; expected "
            f"{TAB_DIRECTION_OUTWARD!r} or {TAB_DIRECTION_INWARD!r}"
        )

    kerf_half = kerf * 0.5

    raw_shapes: dict[int, Polygon] = {}
    exclusion_zones: dict[int, list[Polygon]] = {}
    for fid, proj in projections.items():
        raw_shapes[fid] = _polygon_to_shapely(proj.outer_polygon, proj.inner_polygons)
        exclusion_zones[fid] = _build_exclusion_zones(proj, notch_buffer)

    add_ops: dict[int, list[Polygon]] = {fid: [] for fid in projections}
    sub_ops: dict[int, list[Polygon]] = {fid: [] for fid in projections}

    # Direct shared seams.
    for se in shared_edges:
        fid_a = se.face_a_id
        fid_b = se.face_b_id
        if fid_a not in projections or fid_b not in projections:
            continue

        # Keep deterministic owner ordering. Bottom owns seams touching it.
        if fid_a == bottom_id:
            owner_id, mate_id = fid_a, fid_b
        elif fid_b == bottom_id:
            owner_id, mate_id = fid_b, fid_a
        else:
            owner_id, mate_id = (fid_a, fid_b) if fid_a < fid_b else (fid_b, fid_a)

        owner_proj = projections[owner_id]
        mate_proj = projections[mate_id]
        owner_idx = _find_matching_edge_index(owner_proj, se)
        mate_idx = _find_matching_edge_index(mate_proj, se)
        if owner_idx is None or mate_idx is None:
            continue

        owner_p1, owner_p2 = owner_proj.outer_edges_2d[owner_idx]
        mate_p1, mate_p2 = mate_proj.outer_edges_2d[mate_idx]

        owner_len = _dist_2d(owner_p1, owner_p2)
        mate_len = _dist_2d(mate_p1, mate_p2)
        if owner_len <= 1e-6 or mate_len <= 1e-6:
            continue

        # Conservative terminal margin to avoid corner nibs.
        terminal_margin = max(edge_margin, thickness)

        owner_intervals = _build_owner_intervals(
            owner_len,
            fusion_params,
            start_margin=terminal_margin,
            end_margin=terminal_margin,
        )
        if owner_intervals is None:
            continue

        owner_fingers, owner_slots, owner_lo, owner_hi = owner_intervals

        reversed_edge = _edges_reversed(owner_proj, owner_idx, mate_proj, mate_idx)
        mate_slots = _map_intervals_by_param(owner_len, mate_len, owner_slots, reverse=reversed_edge)
        mate_slots = _clip_intervals_to_terminal_margins(
            mate_len,
            mate_slots,
            start_margin=terminal_margin,
            end_margin=terminal_margin,
        )

        owner_depth = thickness + kerf_half
        mate_depth = max(0.0, thickness - kerf_half)

        owner_out = _outward_direction(owner_p1, owner_p2, raw_shapes[owner_id])
        if tab_direction == TAB_DIRECTION_INWARD:
            owner_in = (-owner_out[0], -owner_out[1])
            owner_notches = _complement_on_range(owner_slots, owner_lo, owner_hi)
            owner_notches = _clip_intervals_to_terminal_margins(
                owner_len,
                owner_notches,
                start_margin=terminal_margin,
                end_margin=terminal_margin,
            )
            owner_polys = _make_comb_from_intervals(
                owner_p1,
                owner_p2,
                owner_depth,
                owner_in,
                owner_notches,
                exclusion_zones=exclusion_zones.get(owner_id, []),
            )
            sub_ops[owner_id].extend(owner_polys)
        else:
            owner_polys = _make_comb_from_intervals(
                owner_p1,
                owner_p2,
                owner_depth,
                owner_out,
                owner_fingers,
                exclusion_zones=exclusion_zones.get(owner_id, []),
            )
            add_ops[owner_id].extend(owner_polys)

        if mate_depth > 1e-9:
            mate_out = _outward_direction(mate_p1, mate_p2, raw_shapes[mate_id])
            mate_in = (-mate_out[0], -mate_out[1])
            mate_polys = _make_comb_from_intervals(
                mate_p1,
                mate_p2,
                mate_depth,
                mate_in,
                mate_slots,
                exclusion_zones=exclusion_zones.get(mate_id, []),
            )
            sub_ops[mate_id].extend(mate_polys)

    # Through seams: non-adjacent walls meeting bottom through slots.
    if faces is not None and bottom_id in projections:
        face_map = {f.face_id: f for f in faces}
        bottom_face = face_map.get(bottom_id)
        if bottom_face is not None:
            bottom_adjacent = set()
            for se in shared_edges:
                if se.face_a_id == bottom_id:
                    bottom_adjacent.add(se.face_b_id)
                elif se.face_b_id == bottom_id:
                    bottom_adjacent.add(se.face_a_id)

            bottom_proj = projections[bottom_id]
            for fid, wall_proj in projections.items():
                if fid == bottom_id or fid in bottom_adjacent:
                    continue
                wall_face = face_map.get(fid)
                if wall_face is None:
                    continue

                endpoints = _find_bottom_edge_endpoints(wall_face, bottom_face)
                if endpoints is None:
                    continue
                p_start_3d, p_end_3d = endpoints

                wall_start = _project_point(
                    p_start_3d,
                    wall_proj.origin_3d,
                    wall_proj.u_axis,
                    wall_proj.v_axis,
                )
                wall_end = _project_point(
                    p_end_3d,
                    wall_proj.origin_3d,
                    wall_proj.u_axis,
                    wall_proj.v_axis,
                )
                slot_start = _project_point(
                    p_start_3d,
                    bottom_proj.origin_3d,
                    bottom_proj.u_axis,
                    bottom_proj.v_axis,
                )
                slot_end = _project_point(
                    p_end_3d,
                    bottom_proj.origin_3d,
                    bottom_proj.u_axis,
                    bottom_proj.v_axis,
                )

                wall_len = _dist_2d(wall_start, wall_end)
                slot_len = _dist_2d(slot_start, slot_end)
                if wall_len <= 1e-6 or slot_len <= 1e-6:
                    continue

                terminal_margin = max(edge_margin, thickness)
                wall_intervals = _build_owner_intervals(
                    wall_len,
                    fusion_params,
                    start_margin=terminal_margin,
                    end_margin=terminal_margin,
                )
                if wall_intervals is None:
                    continue

                wall_fingers, wall_slots, wall_lo, wall_hi = wall_intervals
                bottom_slots = _map_intervals_by_param(wall_len, slot_len, wall_slots, reverse=False)
                bottom_slots = _clip_intervals_to_terminal_margins(
                    slot_len,
                    bottom_slots,
                    start_margin=terminal_margin,
                    end_margin=terminal_margin,
                )

                wall_depth = max(0.0, thickness - kerf_half)
                slot_depth = thickness + kerf_half

                # Bottom receives mapped slots.
                bottom_out = _outward_direction(slot_start, slot_end, raw_shapes[bottom_id])
                bottom_in = (-bottom_out[0], -bottom_out[1])
                bottom_polys = _make_comb_from_intervals(
                    slot_start,
                    slot_end,
                    slot_depth,
                    bottom_in,
                    bottom_slots,
                    exclusion_zones=exclusion_zones.get(bottom_id, []),
                )
                sub_ops[bottom_id].extend(bottom_polys)

                if wall_depth <= 1e-9:
                    continue

                wall_out = _outward_direction(wall_start, wall_end, raw_shapes[fid])
                if tab_direction == TAB_DIRECTION_INWARD:
                    wall_in = (-wall_out[0], -wall_out[1])
                    wall_notches = _complement_on_range(wall_slots, wall_lo, wall_hi)
                    wall_notches = _clip_intervals_to_terminal_margins(
                        wall_len,
                        wall_notches,
                        start_margin=terminal_margin,
                        end_margin=terminal_margin,
                    )
                    wall_polys = _make_comb_from_intervals(
                        wall_start,
                        wall_end,
                        wall_depth,
                        wall_in,
                        wall_notches,
                        exclusion_zones=exclusion_zones.get(fid, []),
                    )
                    sub_ops[fid].extend(wall_polys)
                else:
                    wall_polys = _make_comb_from_intervals(
                        wall_start,
                        wall_end,
                        wall_depth,
                        wall_out,
                        wall_fingers,
                        exclusion_zones=exclusion_zones.get(fid, []),
                    )
                    add_ops[fid].extend(wall_polys)

    modified: dict[int, list[tuple[float, float]]] = {}
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] = {}

    for fid, base in raw_shapes.items():
        shape = base
        if add_ops[fid]:
            u = unary_union(add_ops[fid])
            if not u.is_empty:
                shape = shape.union(u)
        if sub_ops[fid]:
            u = unary_union(sub_ops[fid])
            if not u.is_empty:
                shape = shape.difference(u)

        outer, inners = _shapely_to_vertices(shape)
        modified[fid] = outer
        slot_cutouts[fid] = inners

    return modified, slot_cutouts
