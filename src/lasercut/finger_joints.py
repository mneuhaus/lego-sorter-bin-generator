"""Finger joint pattern generation using Shapely boolean operations.

Instead of manually inserting vertices into polygon arrays, we use Shapely's
robust boolean operations (union/difference) to add tabs and cut slots.
This correctly handles corners, edge transitions, and complex polygon shapes.
"""

import math
from dataclasses import dataclass
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

from .projector import Projection2D, _project_point, _normalize, _cross, _sub, _dot
from .face_classifier import SharedEdge
from .step_loader import PlanarFace, EdgeData

# Buffer distance around inner features (notches/holes) where no fingers are placed.
DEFAULT_NOTCH_BUFFER = 2.0
# Inset from plateau (mountain) boundaries where fingers start/end.
DEFAULT_PLATEAU_INSET = 3.0
# Minimum usable plateau segment length in mm.
DEFAULT_MIN_PLATEAU_LENGTH = 12.0

# Fusion-style overlap slicing defaults
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
    """Sizing/placement settings for the Fusion-style overlap slicing model."""
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
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _polygon_signed_area(points: list[tuple[float, float]]) -> float:
    """Signed area of a polygon ring (positive for CCW winding)."""
    if len(points) < 3:
        return 0.0
    area2 = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return 0.5 * area2


def _lerp_2d(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _corner_endpoint_keepouts(
    proj: Projection2D,
    edge_idx: int,
    depth: float,
    max_factor: float = 2.5,
) -> tuple[float, float]:
    """Compute corner-safe start/end clearances for one outer edge.

    For convex corners, a rectangular tooth of depth `depth` should start/end
    at least `depth * cot(interior_angle/2)` from the vertex to avoid tiny
    miter artifacts where neighboring edge features overlap.
    """
    verts = proj.outer_polygon
    n = len(verts)
    if n < 3 or depth <= 1e-9:
        return 0.0, 0.0

    is_ccw = _polygon_signed_area(verts) >= 0

    def _keepout(prev_pt: tuple[float, float], curr_pt: tuple[float, float], next_pt: tuple[float, float]) -> float:
        in_vec = (curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
        out_vec = (next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])

        in_len = math.hypot(in_vec[0], in_vec[1])
        out_len = math.hypot(out_vec[0], out_vec[1])
        if in_len < 1e-9 or out_len < 1e-9:
            return 0.0

        in_u = (in_vec[0] / in_len, in_vec[1] / in_len)
        out_u = (out_vec[0] / out_len, out_vec[1] / out_len)

        turn = in_u[0] * out_u[1] - in_u[1] * out_u[0]
        is_convex = turn > 1e-9 if is_ccw else turn < -1e-9
        if not is_convex:
            return 0.0

        dot = max(-1.0, min(1.0, in_u[0] * out_u[0] + in_u[1] * out_u[1]))
        interior = math.acos(dot)
        if interior <= 1e-6 or interior >= math.pi - 1e-6:
            return 0.0

        # depth * cot(interior/2)
        half = interior * 0.5
        tan_half = math.tan(half)
        if abs(tan_half) < 1e-9:
            return 0.0

        keepout = depth / tan_half
        if keepout < 0:
            return 0.0
        return min(keepout, depth * max_factor)

    def _keepout_at_vertex(i: int) -> float:
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        return _keepout(verts[i_prev], verts[i], verts[i_next])

    i0 = edge_idx % n
    i1 = (edge_idx + 1) % n
    start_keepout = _keepout_at_vertex(i0)
    end_keepout = _keepout_at_vertex(i1)
    return start_keepout, end_keepout


def _corner_keepouts_near_points(
    proj: Projection2D,
    start_pt: tuple[float, float],
    end_pt: tuple[float, float],
    depth: float,
    tol: float = 6.0,
    max_factor: float = 2.5,
) -> tuple[float, float]:
    """Corner keepouts using nearest outer vertices to two arbitrary endpoints."""
    verts = proj.outer_polygon
    n = len(verts)
    if n < 3 or depth <= 1e-9:
        return 0.0, 0.0

    def nearest_idx(pt: tuple[float, float]) -> int | None:
        best_i = None
        best_d = float("inf")
        for i, v in enumerate(verts):
            d = _dist_2d(pt, v)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is None or best_d > tol:
            return None
        return best_i

    is_ccw = _polygon_signed_area(verts) >= 0

    def keepout_at(i: int) -> float:
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        prev_pt = verts[i_prev]
        curr_pt = verts[i]
        next_pt = verts[i_next]

        in_vec = (curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
        out_vec = (next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
        in_len = math.hypot(in_vec[0], in_vec[1])
        out_len = math.hypot(out_vec[0], out_vec[1])
        if in_len < 1e-9 or out_len < 1e-9:
            return 0.0
        in_u = (in_vec[0] / in_len, in_vec[1] / in_len)
        out_u = (out_vec[0] / out_len, out_vec[1] / out_len)
        turn = in_u[0] * out_u[1] - in_u[1] * out_u[0]
        is_convex = turn > 1e-9 if is_ccw else turn < -1e-9
        if not is_convex:
            return 0.0
        dot = max(-1.0, min(1.0, in_u[0] * out_u[0] + in_u[1] * out_u[1]))
        interior = math.acos(dot)
        if interior <= 1e-6 or interior >= math.pi - 1e-6:
            return 0.0
        tan_half = math.tan(0.5 * interior)
        if abs(tan_half) < 1e-9:
            return 0.0
        return min(max(0.0, depth / tan_half), depth * max_factor)

    start_i = nearest_idx(start_pt)
    end_i = nearest_idx(end_pt)
    start_keepout = keepout_at(start_i) if start_i is not None else 0.0
    end_keepout = keepout_at(end_i) if end_i is not None else 0.0
    return start_keepout, end_keepout


def _find_plateau_segments(p1: tuple[float, float], p2: tuple[float, float],
                           polygon: Polygon, sample_interval: float = 0.5,
                           plateau_inset: float = 3.0) -> list[tuple[float, float]]:
    """Detect plateau (mountain) segments along an edge.

    Samples points along the edge and measures their distance to the polygon
    boundary. Uses an adaptive threshold: if any meaningful variation exists
    (even sub-mm from 3D model notch markers), it detects the pattern.

    Points near the boundary (distance < threshold) are on plateaus
    (the polygon edge coincides with the edge line). Points far from the
    boundary are in valleys (the polygon edge has retreated into a notch).

    Returns parametric ranges (t_start, t_end) with plateau_inset applied at
    each boundary.

    If the entire edge is a plateau (no valleys detected), returns an empty list
    to signal the caller should use the default full-edge distribution.
    """
    edge_len = _dist_2d(p1, p2)
    if edge_len < 1e-6:
        return []

    n_samples = max(20, int(edge_len / sample_interval))
    boundary = polygon.exterior

    # Sample along edge, measuring distance to polygon boundary
    distances = []
    for i in range(n_samples + 1):
        t = i / n_samples
        pt = _lerp_2d(p1, p2, t)
        dist = boundary.distance(Point(pt))
        distances.append(dist)

    max_dist = max(distances)

    # No meaningful variation → all plateau → use default distribution
    if max_dist < 0.01:
        return []

    # Adaptive threshold: use 30% of max observed distance.
    # This detects even tiny 3D-model notch markers (e.g. 0.06mm dips)
    # while correctly classifying mountain vs valley.
    threshold = max_dist * 0.3
    is_plateau = [d < threshold for d in distances]

    # If all samples are plateau, return empty → use default
    if all(is_plateau):
        return []

    # Group contiguous plateau regions
    segments = []
    start_t = None
    for i, val in enumerate(is_plateau):
        t = i / n_samples
        if val and start_t is None:
            start_t = t
        elif not val and start_t is not None:
            end_t = (i - 1) / n_samples
            segments.append((start_t, end_t))
            start_t = None
    if start_t is not None:
        segments.append((start_t, 1.0))

    # Apply inset from plateau boundaries
    inset_t = plateau_inset / edge_len
    result = []
    for s, e in segments:
        s2 = s + inset_t
        e2 = e - inset_t
        if e2 > s2 + 1e-6:
            result.append((s2, e2))

    return result


def _intersect_segment_lists(segs_a: list[tuple[float, float]],
                             segs_b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Intersect two sorted lists of (start, end) parametric segments."""
    result = []
    i, j = 0, 0
    while i < len(segs_a) and j < len(segs_b):
        a_s, a_e = segs_a[i]
        b_s, b_e = segs_b[j]
        start = max(a_s, b_s)
        end = min(a_e, b_e)
        if end > start + 1e-6:
            result.append((start, end))
        if a_e < b_e:
            i += 1
        else:
            j += 1
    return result


def _reverse_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Reverse parametric segments (t → 1-t) and re-sort."""
    return sorted([(1.0 - e, 1.0 - s) for s, e in segments])


def _edges_reversed(proj_a: 'Projection2D', edge_idx_a: int,
                    proj_b: 'Projection2D', edge_idx_b: int,
                    tol: float = 0.5) -> bool:
    """Check if two 2D edges correspond to reversed 3D edges."""
    ea = proj_a.edge_map_3d[edge_idx_a]
    eb = proj_b.edge_map_3d[edge_idx_b]

    fwd = (all(abs(a - b) < tol for a, b in zip(ea.start, eb.start)) and
           all(abs(a - b) < tol for a, b in zip(ea.end, eb.end)))
    rev = (all(abs(a - b) < tol for a, b in zip(ea.start, eb.end)) and
           all(abs(a - b) < tol for a, b in zip(ea.end, eb.start)))

    return rev and not fwd


def _polygon_to_shapely(outer: list[tuple[float, float]],
                        holes: list[list[tuple[float, float]]] | None = None) -> Polygon:
    """Convert vertex lists to a Shapely Polygon."""
    if len(outer) < 3:
        return Polygon()
    hole_rings = [h for h in (holes or []) if len(h) >= 3]
    return Polygon(outer, hole_rings)


def _shapely_to_vertices(geom) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """Convert a Shapely geometry back to outer + inner vertex lists."""
    if geom.is_empty:
        return [], []

    if isinstance(geom, MultiPolygon):
        # Take the largest polygon
        geom = max(geom.geoms, key=lambda g: g.area)
    elif isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return [], []
        geom = max(polys, key=lambda g: g.area)

    outer = list(geom.exterior.coords[:-1])  # Shapely repeats the first point
    inners = [list(ring.coords[:-1]) for ring in geom.interiors]
    return outer, inners


def _find_matching_edge_index(proj: Projection2D, shared_edge: SharedEdge) -> int | None:
    """Find which 2D edge in the projection corresponds to the shared 3D edge."""
    tol = 0.5
    for idx, edge_3d in enumerate(proj.edge_map_3d):
        for se in [shared_edge.edge_a, shared_edge.edge_b]:
            mid_match = all(abs(a - b) < tol for a, b in zip(edge_3d.midpoint, se.midpoint))
            if mid_match:
                return idx

            fwd = (all(abs(a - b) < tol for a, b in zip(edge_3d.start, se.start)) and
                   all(abs(a - b) < tol for a, b in zip(edge_3d.end, se.end)))
            rev = (all(abs(a - b) < tol for a, b in zip(edge_3d.start, se.end)) and
                   all(abs(a - b) < tol for a, b in zip(edge_3d.end, se.start)))
            if fwd or rev:
                return idx

    return None


def _find_edge_index_by_endpoints(
    proj: Projection2D,
    p1: tuple[float, float],
    p2: tuple[float, float],
    tol: float = 0.5,
) -> int | None:
    """Find a projected outer edge index by 2D endpoints (either orientation)."""
    for idx, (a, b) in enumerate(proj.outer_edges_2d):
        fwd = _dist_2d(a, p1) <= tol and _dist_2d(b, p2) <= tol
        rev = _dist_2d(a, p2) <= tol and _dist_2d(b, p1) <= tol
        if fwd or rev:
            return idx
    return None




def _outward_direction(p1: tuple[float, float], p2: tuple[float, float],
                       polygon: Polygon) -> tuple[float, float]:
    """Compute the outward-pointing unit perpendicular from edge p1->p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-12:
        return (0.0, 1.0)

    # Two candidate normals
    n1 = (-dy / length, dx / length)
    n2 = (dy / length, -dx / length)

    mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    centroid = polygon.centroid

    test1 = (mid[0] + n1[0], mid[1] + n1[1])
    test2 = (mid[0] + n2[0], mid[1] + n2[1])

    d1 = (test1[0] - centroid.x)**2 + (test1[1] - centroid.y)**2
    d2 = (test2[0] - centroid.x)**2 + (test2[1] - centroid.y)**2

    return n1 if d1 > d2 else n2



def _find_bottom_edge_endpoints(wall: PlanarFace, bottom: PlanarFace, tol: float = 5.0):
    """Find the endpoints of a wall's bottom edge (nearest the bottom plate plane)."""
    bn = _normalize(bottom.normal)

    def plane_dist(pt):
        return _dot(bn, _sub(pt, bottom.center))

    edge_dists = [(abs(plane_dist(e.midpoint)), e) for e in wall.outer_wire_edges]
    min_dist = min(d for d, _ in edge_dists)
    bottom_edges = [(d, e) for d, e in edge_dists if d < min_dist + tol]

    if not bottom_edges:
        return None

    all_pts = [e.start for _, e in bottom_edges] + [e.end for _, e in bottom_edges]

    max_d = 0
    best_pair = None
    for i, p1 in enumerate(all_pts):
        for j, p2 in enumerate(all_pts):
            if j <= i:
                continue
            d = sum((a - b)**2 for a, b in zip(p1, p2))**0.5
            if d > max_d:
                max_d = d
                best_pair = (p1, p2)

    return best_pair


def _build_exclusion_zones(proj: Projection2D, buffer_dist: float) -> list[Polygon]:
    """Build exclusion zones from a face's inner polygons (notches, holes).

    Each inner polygon is buffered outward to create a safe zone where
    no fingers should be placed.
    """
    zones = []
    for inner in proj.inner_polygons:
        if len(inner) >= 3:
            poly = Polygon(inner)
            if poly.is_valid and not poly.is_empty:
                buffered = poly.buffer(buffer_dist)
                if not buffered.is_empty:
                    zones.append(buffered)
    return zones



def _define_fusion_intervals(
    size: float,
    params: FusionJointParams,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    """Port of Fusion add-in interval math.

    Returns:
      (finger_intervals, slot_intervals)
      finger_intervals: where tabs are present
      slot_intervals: where mating slots are cut (includes gap on both sides)
    """
    placement_type = params.placement_type
    dynamic_size_type = params.dynamic_size_type
    min_finger_size = params.min_finger_size
    min_notch_size = params.min_notch_size
    fixed_notch_size = params.fixed_notch_size
    fixed_finger_size = params.fixed_finger_size
    is_number_of_fingers_fixed = params.is_number_of_fingers_fixed
    fixed_num_fingers = params.fixed_num_fingers
    gap_size = params.gap

    if size <= 0:
        return None

    if is_number_of_fingers_fixed:
        num_fingers = max(1, int(fixed_num_fingers))
        if placement_type == FUSION_PLACEMENT_FINGERS_OUTSIDE:
            num_notches = num_fingers - 1
        elif placement_type == FUSION_PLACEMENT_NOTCHES_OUTSIDE:
            num_notches = num_fingers + 1
        else:
            num_notches = num_fingers

        num_gaps = num_fingers + num_notches - 1
        total_gap_size = num_gaps * gap_size

        if dynamic_size_type == FUSION_DYNAMIC_EQUAL:
            denom = num_fingers + num_notches
            if denom <= 0:
                return None
            finger_size = (size - total_gap_size) / denom
            notch_size = finger_size
        elif dynamic_size_type == FUSION_DYNAMIC_FIXED_NOTCH:
            notch_size = fixed_notch_size
            if num_fingers <= 0:
                return None
            finger_size = (size - total_gap_size - num_notches * notch_size) / num_fingers
        elif dynamic_size_type == FUSION_DYNAMIC_FIXED_FINGER:
            finger_size = fixed_finger_size
            if num_notches <= 0:
                return None
            notch_size = (size - total_gap_size - num_fingers * finger_size) / num_notches
        else:
            return None
    else:
        if dynamic_size_type == FUSION_DYNAMIC_EQUAL:
            denom = min_finger_size + gap_size
            if denom <= 0:
                return None
            max_num = int((size + gap_size) / denom)
            num_fingers = num_notches = int(max_num / 2)
            if placement_type == FUSION_PLACEMENT_FINGERS_OUTSIDE:
                if max_num % 2 == 1:
                    num_fingers += 1
                else:
                    num_notches -= 1
            elif placement_type == FUSION_PLACEMENT_NOTCHES_OUTSIDE:
                if max_num % 2 == 1:
                    num_notches += 1
                else:
                    num_fingers -= 1

            if num_fingers + num_notches == 0:
                return None
            num_gaps = num_fingers + num_notches - 1
            total_gap_size = num_gaps * gap_size
            finger_size = (size - total_gap_size) / (num_fingers + num_notches)
            notch_size = finger_size
        elif dynamic_size_type == FUSION_DYNAMIC_FIXED_NOTCH:
            notch_size = fixed_notch_size
            extra_notch = 0
            if placement_type == FUSION_PLACEMENT_FINGERS_OUTSIDE:
                extra_notch = -1
            elif placement_type == FUSION_PLACEMENT_NOTCHES_OUTSIDE:
                extra_notch = 1
            denom = notch_size + min_finger_size + 2 * gap_size
            if denom <= 0:
                return None
            num_fingers = int((size - extra_notch * (notch_size + gap_size) + gap_size) / denom)
            num_notches = num_fingers + extra_notch
            if num_fingers == 0:
                return None
            num_gaps = num_fingers + num_notches - 1
            total_gap_size = num_gaps * gap_size
            finger_size = (size - total_gap_size - num_notches * notch_size) / num_fingers
        elif dynamic_size_type == FUSION_DYNAMIC_FIXED_FINGER:
            finger_size = fixed_finger_size
            extra_finger = 0
            if placement_type == FUSION_PLACEMENT_FINGERS_OUTSIDE:
                extra_finger = 1
            elif placement_type == FUSION_PLACEMENT_NOTCHES_OUTSIDE:
                extra_finger = -1
            denom = finger_size + min_notch_size + 2 * gap_size
            if denom <= 0:
                return None
            num_notches = int((size - extra_finger * (finger_size + gap_size) + gap_size) / denom)
            num_fingers = num_notches + extra_finger
            if num_notches == 0:
                return None
            num_gaps = num_fingers + num_notches - 1
            total_gap_size = num_gaps * gap_size
            notch_size = (size - total_gap_size - num_fingers * finger_size) / num_notches
        else:
            return None

    epsilon = 1e-5
    if (
        num_fingers < 0
        or num_notches < 0
        or finger_size <= epsilon
        or notch_size <= epsilon
    ):
        return None

    consumed = (
        finger_size * num_fingers
        + notch_size * num_notches
        + (num_fingers + num_notches - 1) * gap_size
    )
    if consumed - epsilon > size:
        return None

    if placement_type in [FUSION_PLACEMENT_FINGERS_OUTSIDE, FUSION_PLACEMENT_SAME_START_FINGER]:
        finger_start = 0.0
        notch_start = finger_size + gap_size
    else:
        finger_start = notch_size + gap_size
        notch_start = 0.0

    spacing = finger_size + notch_size + 2 * gap_size
    finger_intervals = [
        (finger_start + i * spacing, finger_size)
        for i in range(num_fingers)
    ]
    # Fusion behavior: tool body includes the full gap on both sides.
    slot_intervals = [
        (finger_start + i * spacing - gap_size, finger_size + 2 * gap_size)
        for i in range(num_fingers)
    ]
    # Keep notch_start alive for parity with the source formulas and easier debugging.
    _ = notch_start
    return finger_intervals, slot_intervals


def _build_fusion_intervals_for_segments(
    edge_len: float,
    params: FusionJointParams,
    segments_t: list[tuple[float, float]] | None = None,
    margin: float = 0.0,
    start_margin: float | None = None,
    end_margin: float | None = None,
    min_segment_length: float = 0.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    """Build Fusion-style intervals on one or more parametric edge segments."""
    segments = _resolve_joint_segments(
        edge_len=edge_len,
        segments_t=segments_t,
        margin=margin,
        start_margin=start_margin,
        end_margin=end_margin,
        min_segment_length=min_segment_length,
    )
    if segments is None:
        return None

    finger_intervals: list[tuple[float, float]] = []
    slot_intervals: list[tuple[float, float]] = []
    for s, e in segments:
        seg_len = (e - s) * edge_len
        seg_intervals = _define_fusion_intervals(seg_len, params)
        if seg_intervals is None:
            continue
        seg_fingers, seg_slots = seg_intervals
        offset = s * edge_len
        finger_intervals.extend([(offset + a, w) for a, w in seg_fingers])
        slot_intervals.extend([(offset + a, w) for a, w in seg_slots])

    if not finger_intervals and not slot_intervals:
        return None
    return sorted(finger_intervals), sorted(slot_intervals)


def _resolve_joint_segments(
    edge_len: float,
    segments_t: list[tuple[float, float]] | None = None,
    margin: float = 0.0,
    start_margin: float | None = None,
    end_margin: float | None = None,
    min_segment_length: float = 0.0,
) -> list[tuple[float, float]] | None:
    """Resolve valid parametric edge segments after applying margins."""
    if edge_len <= 1e-9:
        return None

    start_m = margin if start_margin is None else start_margin
    end_m = margin if end_margin is None else end_margin
    start_t = max(0.0, start_m / edge_len) if start_m > 0 else 0.0
    end_t = max(0.0, end_m / edge_len) if end_m > 0 else 0.0
    start_t = min(start_t, 0.49)
    end_t = min(end_t, 0.49)
    if start_t + end_t >= 1.0 - 1e-6:
        return None

    if segments_t:
        segments: list[tuple[float, float]] = []
        for s, e in segments_t:
            s2 = max(s, start_t)
            e2 = min(e, 1.0 - end_t)
            seg_len = (e2 - s2) * edge_len
            if seg_len >= max(1e-6, min_segment_length):
                segments.append((s2, e2))
        return segments

    return [(start_t, 1.0 - end_t)]


def _placement_edge_notch_policy(placement_type: str) -> tuple[bool, bool]:
    """Whether the placement mode should include notch regions at segment edges."""
    if placement_type == FUSION_PLACEMENT_NOTCHES_OUTSIDE:
        return True, True
    if placement_type == FUSION_PLACEMENT_SAME_START_FINGER:
        return False, True
    if placement_type == FUSION_PLACEMENT_SAME_START_NOTCH:
        return True, False
    return False, False


def _notch_intervals_from_fingers(
    finger_intervals: list[tuple[float, float]],
    seg_len: float,
    include_start: bool,
    include_end: bool,
) -> list[tuple[float, float]]:
    """Derive notch intervals from finger spans on one local segment."""
    eps = 1e-6
    if seg_len <= eps:
        return []

    clipped: list[tuple[float, float]] = []
    for start, width in finger_intervals:
        if width <= eps:
            continue
        s = max(0.0, start)
        e = min(seg_len, start + width)
        if e > s + eps:
            clipped.append((s, e))
    clipped.sort()
    if not clipped:
        return [(0.0, seg_len)] if include_start and include_end else []

    notches: list[tuple[float, float]] = []

    first_start, first_end = clipped[0]
    if include_start and first_start > eps:
        notches.append((0.0, first_start))

    prev_end = first_end
    for start, end in clipped[1:]:
        gap_width = start - prev_end
        if gap_width > eps:
            notches.append((prev_end, gap_width))
        prev_end = max(prev_end, end)

    if include_end and seg_len - prev_end > eps:
        notches.append((prev_end, seg_len - prev_end))

    return notches


def _build_inward_notch_intervals_for_segments(
    edge_len: float,
    params: FusionJointParams,
    segments_t: list[tuple[float, float]] | None = None,
    margin: float = 0.0,
    start_margin: float | None = None,
    end_margin: float | None = None,
    min_segment_length: float = 0.0,
    force_terminal_margin_notches: bool = False,
) -> list[tuple[float, float]]:
    """Build inward-notch intervals per segment to avoid cross-segment artifacts."""
    eps = 1e-6
    segments = _resolve_joint_segments(
        edge_len=edge_len,
        segments_t=segments_t,
        margin=margin,
        start_margin=start_margin,
        end_margin=end_margin,
        min_segment_length=min_segment_length,
    )
    if not segments:
        return []

    include_start, include_end = _placement_edge_notch_policy(params.placement_type)
    notch_intervals: list[tuple[float, float]] = []
    for s, e in segments:
        seg_len = (e - s) * edge_len
        seg_intervals = _define_fusion_intervals(seg_len, params)
        if seg_intervals is None:
            continue
        seg_fingers, _ = seg_intervals
        seg_notches = _notch_intervals_from_fingers(
            seg_fingers,
            seg_len=seg_len,
            include_start=include_start,
            include_end=include_end,
        )
        offset = s * edge_len
        notch_intervals.extend((offset + start, width) for start, width in seg_notches)

    # For placements that start/end with notches (or when explicitly forced),
    # treat terminal margins outside the active segment range as notches.
    # This avoids unmatched end-fingers in inward mode.
    if segments:
        first_seg_start = segments[0][0] * edge_len
        last_seg_end = segments[-1][1] * edge_len
        if (include_start or force_terminal_margin_notches) and first_seg_start > eps:
            notch_intervals.append((0.0, first_seg_start))
        end_leftover = edge_len - last_seg_end
        if (include_end or force_terminal_margin_notches) and end_leftover > eps:
            notch_intervals.append((last_seg_end, end_leftover))

    if not notch_intervals:
        return []

    merged: list[tuple[float, float]] = []
    for start, width in sorted(notch_intervals):
        if width <= eps:
            continue
        end = start + width
        if not merged:
            merged.append((start, width))
            continue
        prev_start, prev_width = merged[-1]
        prev_end = prev_start + prev_width
        if start <= prev_end + eps:
            merged[-1] = (prev_start, max(prev_end, end) - prev_start)
        else:
            merged.append((start, width))
    return merged


def _make_comb_from_intervals(
    p1: tuple[float, float],
    p2: tuple[float, float],
    depth: float,
    outward_dir: tuple[float, float],
    intervals: list[tuple[float, float]],
    start_offset: float = 0.0,
    overlap: float = 0.01,
    exclusion_zones: list[Polygon] | None = None,
) -> list[Polygon]:
    """Create rectangular teeth from absolute intervals along an edge."""
    teeth: list[Polygon] = []
    edge_len = _dist_2d(p1, p2)
    if edge_len < 1e-6 or depth <= 1e-9:
        return teeth

    ux = (p2[0] - p1[0]) / edge_len
    uy = (p2[1] - p1[1]) / edge_len
    nx, ny = outward_dir

    for start, width in intervals:
        if width <= 1e-9:
            continue
        a = start_offset + start
        b = a + width
        if b <= a + 1e-9:
            continue

        s0 = (p1[0] + ux * a, p1[1] + uy * a)
        s1 = (p1[0] + ux * b, p1[1] + uy * b)

        rect = Polygon([
            (s0[0] - nx * overlap, s0[1] - ny * overlap),
            (s1[0] - nx * overlap, s1[1] - ny * overlap),
            (s1[0] + nx * depth, s1[1] + ny * depth),
            (s0[0] + nx * depth, s0[1] + ny * depth),
        ])
        if not rect.is_valid or rect.area <= 0:
            continue
        geom = rect
        if exclusion_zones:
            clipped = rect.difference(unary_union(exclusion_zones))
            if clipped.is_empty:
                continue
            geom = clipped

        if isinstance(geom, Polygon):
            candidates = [geom]
        elif isinstance(geom, MultiPolygon):
            candidates = list(geom.geoms)
        elif isinstance(geom, GeometryCollection):
            candidates = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        else:
            candidates = []

        for poly in candidates:
            if poly.is_valid and poly.area > 1e-9:
                teeth.append(poly)

    return teeth


def _complement_notch_intervals(
    edge_len: float,
    finger_intervals: list[tuple[float, float]],
    eps: float = 1e-6,
) -> list[tuple[float, float]]:
    """Return notch intervals as the complement of finger intervals on [0, edge_len]."""
    if edge_len <= eps or not finger_intervals:
        return []

    spans: list[tuple[float, float]] = []
    for start, width in finger_intervals:
        if width <= eps:
            continue
        s = max(0.0, start)
        e = min(edge_len, start + width)
        if e > s + eps:
            spans.append((s, e))
    if not spans:
        return []

    spans.sort()
    merged: list[tuple[float, float]] = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe + eps:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    notches: list[tuple[float, float]] = []
    cur = 0.0
    for s, e in merged:
        if s > cur + eps:
            notches.append((cur, s - cur))
        cur = max(cur, e)
    if edge_len > cur + eps:
        notches.append((cur, edge_len - cur))
    return notches


def _clip_intervals_to_terminal_margins(
    edge_len: float,
    intervals: list[tuple[float, float]],
    start_margin: float = 0.0,
    end_margin: float = 0.0,
    eps: float = 1e-6,
) -> list[tuple[float, float]]:
    """Clip intervals to [start_margin, edge_len - end_margin] and merge."""
    if edge_len <= eps or not intervals:
        return []

    lo = max(0.0, start_margin)
    hi = edge_len - max(0.0, end_margin)
    if hi <= lo + eps:
        return []

    clipped: list[tuple[float, float]] = []
    for start, width in intervals:
        if width <= eps:
            continue
        s = max(lo, start)
        e = min(hi, start + width)
        if e > s + eps:
            clipped.append((s, e - s))

    if not clipped:
        return []

    merged: list[tuple[float, float]] = []
    for start, width in sorted(clipped):
        end = start + width
        if not merged:
            merged.append((start, width))
            continue
        ps, pw = merged[-1]
        pe = ps + pw
        if start <= pe + eps:
            merged[-1] = (ps, max(pe, end) - ps)
        else:
            merged.append((start, width))
    return merged


def _map_intervals_by_param(
    src_len: float,
    dst_len: float,
    intervals: list[tuple[float, float]],
    reverse: bool = False,
    eps: float = 1e-6,
) -> list[tuple[float, float]]:
    """Map absolute intervals from one edge parameterization to another."""
    if src_len <= eps or dst_len <= eps or not intervals:
        return []

    mapped: list[tuple[float, float]] = []
    for start, width in intervals:
        if width <= eps:
            continue
        s = max(0.0, start)
        e = min(src_len, start + width)
        if e <= s + eps:
            continue

        t0 = s / src_len
        t1 = e / src_len
        if reverse:
            m0 = dst_len * (1.0 - t1)
            m1 = dst_len * (1.0 - t0)
        else:
            m0 = dst_len * t0
            m1 = dst_len * t1

        if m1 > m0 + eps:
            mapped.append((m0, m1 - m0))

    if not mapped:
        return []

    merged: list[tuple[float, float]] = []
    for start, width in sorted(mapped):
        end = start + width
        if not merged:
            merged.append((start, width))
            continue
        ps, pw = merged[-1]
        pe = ps + pw
        if start <= pe + eps:
            merged[-1] = (ps, max(pe, end) - ps)
        else:
            merged.append((start, width))
    return merged


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
    """Apply joints using a Fusion-style overlap slicing interval model."""
    if fusion_params is None:
        fusion_params = FusionJointParams()

    # Fusion add-in has no explicit edge margin concept; default to 0 unless set.
    if edge_margin < 0:
        edge_margin = FUSION_DEFAULT_EDGE_MARGIN
    if notch_buffer < 0:
        notch_buffer = DEFAULT_NOTCH_BUFFER
    if plateau_inset < 0:
        plateau_inset = DEFAULT_PLATEAU_INSET
    if min_plateau_length < 0:
        min_plateau_length = DEFAULT_MIN_PLATEAU_LENGTH

    if tab_direction not in (TAB_DIRECTION_OUTWARD, TAB_DIRECTION_INWARD):
        raise ValueError(
            f"Unsupported tab_direction={tab_direction!r}; expected "
            f"{TAB_DIRECTION_OUTWARD!r} or {TAB_DIRECTION_INWARD!r}"
        )

    kerf_half = kerf / 2.0
    raw_shapes: dict[int, Polygon] = {}
    shapes: dict[int, Polygon] = {}
    for fid, proj in projections.items():
        shape = _polygon_to_shapely(proj.outer_polygon, proj.inner_polygons)
        raw_shapes[fid] = shape
        # Morphological cleanup is intentionally disabled by default.
        # Earlier cleanup passes (buffer/unbuffer) could introduce fake radii
        # or flatten true STEP radii, causing dimension-debug overlay drift.
        close_dist = 0.0
        if close_dist > 1e-9:
            if fid == bottom_id:
                cleaned = shape.buffer(close_dist, join_style='mitre').buffer(-close_dist, join_style='mitre')
            else:
                cleaned = shape.buffer(-close_dist, join_style='mitre').buffer(close_dist, join_style='mitre')
            if not cleaned.is_empty and cleaned.is_valid:
                if isinstance(cleaned, MultiPolygon):
                    cleaned = max(cleaned.geoms, key=lambda g: g.area)
                shape = cleaned
        shapes[fid] = shape

    exclusion_zones: dict[int, list[Polygon]] = {}
    for fid, proj in projections.items():
        exclusion_zones[fid] = _build_exclusion_zones(proj, notch_buffer)

    # Collect edge operations first, then apply once per face. This keeps
    # edge contracts deterministic and avoids sequential edge-order drift.
    add_ops: dict[int, list[Polygon]] = {fid: [] for fid in projections}
    sub_ops: dict[int, list[Polygon]] = {fid: [] for fid in projections}

    # Direct shared-edge joints: compute intervals once on the owner side and
    # map them to the mate edge, instead of recomputing independently.
    for se in shared_edges:
        fid_a = se.face_a_id
        fid_b = se.face_b_id
        if fid_a not in projections or fid_b not in projections:
            continue

        if fid_a == bottom_id:
            pos_id, neg_id = fid_a, fid_b
        elif fid_b == bottom_id:
            pos_id, neg_id = fid_b, fid_a
        else:
            pos_id = min(fid_a, fid_b)
            neg_id = max(fid_a, fid_b)

        pos_proj = projections[pos_id]
        neg_proj = projections[neg_id]
        pos_edge_idx = _find_matching_edge_index(pos_proj, se)
        neg_edge_idx = _find_matching_edge_index(neg_proj, se)
        if pos_edge_idx is None or neg_edge_idx is None:
            continue

        pos_p1, pos_p2 = pos_proj.outer_edges_2d[pos_edge_idx]
        neg_p1, neg_p2 = neg_proj.outer_edges_2d[neg_edge_idx]
        pos_len = _dist_2d(pos_p1, pos_p2)
        neg_len = _dist_2d(neg_p1, neg_p2)
        pos_depth = thickness + kerf_half
        neg_depth = thickness - kerf_half
        pos_start_keepout, pos_end_keepout = _corner_endpoint_keepouts(
            pos_proj, pos_edge_idx, max(pos_depth, 0.0)
        )
        neg_start_keepout, neg_end_keepout = _corner_endpoint_keepouts(
            neg_proj, neg_edge_idx, max(neg_depth, 0.0)
        )

        pos_plateaus = _find_plateau_segments(
            pos_p1, pos_p2, raw_shapes[pos_id], plateau_inset=plateau_inset
        )
        neg_plateaus = _find_plateau_segments(
            neg_p1, neg_p2, raw_shapes[neg_id], plateau_inset=plateau_inset
        )
        reversed_edge = _edges_reversed(pos_proj, pos_edge_idx, neg_proj, neg_edge_idx)

        if pos_plateaus and neg_plateaus:
            neg_in_pos = _reverse_segments(neg_plateaus) if reversed_edge else neg_plateaus
            shared_pos = _intersect_segment_lists(pos_plateaus, neg_in_pos)
        elif pos_plateaus:
            shared_pos = pos_plateaus
        elif neg_plateaus:
            shared_pos = _reverse_segments(neg_plateaus) if reversed_edge else neg_plateaus
        else:
            shared_pos = []

        # Keep mating faces in lock-step: shared margins are computed in the
        # positive face's parametric direction and mapped to the negative side.
        if reversed_edge:
            neg_start_in_pos = neg_end_keepout
            neg_end_in_pos = neg_start_keepout
        else:
            neg_start_in_pos = neg_start_keepout
            neg_end_in_pos = neg_end_keepout
        shared_start_margin = max(edge_margin, pos_start_keepout, neg_start_in_pos)
        shared_end_margin = max(edge_margin, pos_end_keepout, neg_end_in_pos)

        owner_intervals = _build_fusion_intervals_for_segments(
            pos_len,
            fusion_params,
            segments_t=shared_pos,
            margin=edge_margin,
            start_margin=shared_start_margin,
            end_margin=shared_end_margin,
            min_segment_length=min_plateau_length,
        )
        if owner_intervals is None:
            continue
        pos_finger_intervals, owner_slot_intervals = owner_intervals
        neg_slot_intervals = _map_intervals_by_param(
            pos_len,
            neg_len,
            owner_slot_intervals,
            reverse=reversed_edge,
        )

        # Positive face geometry: tabs outward (legacy) or notches inward.
        outward_pos = _outward_direction(pos_p1, pos_p2, raw_shapes[pos_id])
        if tab_direction == TAB_DIRECTION_INWARD:
            inward_pos = (-outward_pos[0], -outward_pos[1])
            # Derive inward notches from the same owner slot contract that
            # drives mate cutouts, then clip terminal margins to avoid corner
            # endpoint artifacts.
            pos_notch_intervals = _complement_notch_intervals(pos_len, owner_slot_intervals)
            pos_notch_intervals = _clip_intervals_to_terminal_margins(
                pos_len,
                pos_notch_intervals,
                start_margin=shared_start_margin,
                end_margin=shared_end_margin,
            )
            pos_notches = _make_comb_from_intervals(
                pos_p1, pos_p2, pos_depth, inward_pos, pos_notch_intervals,
                exclusion_zones=exclusion_zones.get(pos_id, []),
            )
            sub_ops[pos_id].extend(pos_notches)
        else:
            pos_teeth = _make_comb_from_intervals(
                pos_p1, pos_p2, pos_depth, outward_pos, pos_finger_intervals,
                exclusion_zones=exclusion_zones.get(pos_id, []),
            )
            add_ops[pos_id].extend(pos_teeth)

        # Mating slots on negative face
        if neg_depth > 1e-9:
            outward_neg = _outward_direction(neg_p1, neg_p2, raw_shapes[neg_id])
            inward_neg = (-outward_neg[0], -outward_neg[1])
            neg_teeth = _make_comb_from_intervals(
                neg_p1, neg_p2, neg_depth, inward_neg, neg_slot_intervals,
                exclusion_zones=exclusion_zones.get(neg_id, []),
            )
            sub_ops[neg_id].extend(neg_teeth)

    # Through-slot joints for walls not directly adjacent to bottom
    if faces is not None:
        face_map = {f.face_id: f for f in faces}
        bottom_face = face_map.get(bottom_id)
        if bottom_face is not None:
            bottom_adjacent = set()
            for se in shared_edges:
                if se.face_a_id == bottom_id:
                    bottom_adjacent.add(se.face_b_id)
                elif se.face_b_id == bottom_id:
                    bottom_adjacent.add(se.face_a_id)

            for fid, proj in projections.items():
                if fid == bottom_id or fid in bottom_adjacent:
                    continue

                wall_face = face_map.get(fid)
                if wall_face is None:
                    continue
                endpoints = _find_bottom_edge_endpoints(wall_face, bottom_face)
                if endpoints is None:
                    continue
                p_start_3d, p_end_3d = endpoints

                bottom_proj = projections[bottom_id]
                slot_start = _project_point(
                    p_start_3d, bottom_proj.origin_3d, bottom_proj.u_axis, bottom_proj.v_axis
                )
                slot_end = _project_point(
                    p_end_3d, bottom_proj.origin_3d, bottom_proj.u_axis, bottom_proj.v_axis
                )
                wall_proj = projections[fid]
                wall_start = _project_point(
                    p_start_3d, wall_proj.origin_3d, wall_proj.u_axis, wall_proj.v_axis
                )
                wall_end = _project_point(
                    p_end_3d, wall_proj.origin_3d, wall_proj.u_axis, wall_proj.v_axis
                )

                slot_len = _dist_2d(slot_start, slot_end)
                wall_len = _dist_2d(wall_start, wall_end)
                slot_depth = thickness + kerf_half
                tab_depth = thickness - kerf_half

                wall_plateaus = _find_plateau_segments(
                    wall_start, wall_end, raw_shapes[fid], plateau_inset=plateau_inset
                )
                bottom_plateaus = _find_plateau_segments(
                    slot_start, slot_end, raw_shapes[bottom_id], plateau_inset=plateau_inset
                )

                if wall_plateaus and bottom_plateaus:
                    intersection = _intersect_segment_lists(wall_plateaus, bottom_plateaus)
                    intersection = [
                        (s, e) for s, e in intersection
                        if (e - s) * wall_len >= min_plateau_length - 1e-6
                    ]
                    shared_plateaus = intersection if intersection else wall_plateaus
                elif wall_plateaus:
                    shared_plateaus = wall_plateaus
                elif bottom_plateaus:
                    shared_plateaus = bottom_plateaus
                else:
                    shared_plateaus = []

                wall_edge_idx = _find_edge_index_by_endpoints(wall_proj, wall_start, wall_end)
                if wall_edge_idx is not None:
                    wall_start_keepout, wall_end_keepout = _corner_endpoint_keepouts(
                        wall_proj, wall_edge_idx, max(tab_depth, 0.0)
                    )
                else:
                    wall_start_keepout, wall_end_keepout = _corner_keepouts_near_points(
                        wall_proj, wall_start, wall_end, max(tab_depth, 0.0)
                    )
                bottom_start_keepout, bottom_end_keepout = _corner_keepouts_near_points(
                    bottom_proj, slot_start, slot_end, max(slot_depth, 0.0)
                )
                shared_start_margin = max(edge_margin, wall_start_keepout, bottom_start_keepout)
                shared_end_margin = max(edge_margin, wall_end_keepout, bottom_end_keepout)

                # Owner contract is the wall edge; bottom slots are mapped from
                # owner intervals to keep both sides in lock-step.
                wall_intervals = _build_fusion_intervals_for_segments(
                    wall_len,
                    fusion_params,
                    segments_t=shared_plateaus,
                    margin=edge_margin,
                    start_margin=shared_start_margin,
                    end_margin=shared_end_margin,
                    min_segment_length=min_plateau_length,
                )
                if wall_intervals is None:
                    continue
                finger_intervals, wall_slot_intervals = wall_intervals
                slot_intervals = _map_intervals_by_param(
                    wall_len,
                    slot_len,
                    wall_slot_intervals,
                    reverse=False,
                )

                # Slots on bottom plate
                outward_bottom = _outward_direction(slot_start, slot_end, raw_shapes[bottom_id])
                inward_bottom = (-outward_bottom[0], -outward_bottom[1])
                bottom_slots = _make_comb_from_intervals(
                    slot_start, slot_end, slot_depth, inward_bottom, slot_intervals,
                    exclusion_zones=exclusion_zones.get(bottom_id, []),
                )
                sub_ops[bottom_id].extend(bottom_slots)

                # Wall geometry: tabs outward (legacy) or notches inward.
                if tab_depth <= 1e-9:
                    continue
                outward_wall = _outward_direction(wall_start, wall_end, raw_shapes[fid])
                if tab_direction == TAB_DIRECTION_INWARD:
                    inward_wall = (-outward_wall[0], -outward_wall[1])
                    # Use slot-based complement for through walls as well so
                    # both sides stay in lock-step; clip terminal margins for
                    # corner safety.
                    wall_notch_intervals = _complement_notch_intervals(wall_len, wall_slot_intervals)
                    wall_notch_intervals = _clip_intervals_to_terminal_margins(
                        wall_len,
                        wall_notch_intervals,
                        start_margin=shared_start_margin,
                        end_margin=shared_end_margin,
                    )
                    wall_notches = _make_comb_from_intervals(
                        wall_start, wall_end, tab_depth, inward_wall, wall_notch_intervals,
                        exclusion_zones=exclusion_zones.get(fid, []),
                    )
                    sub_ops[fid].extend(wall_notches)
                else:
                    wall_tabs = _make_comb_from_intervals(
                        wall_start, wall_end, tab_depth, outward_wall, finger_intervals,
                        exclusion_zones=exclusion_zones.get(fid, []),
                    )
                    add_ops[fid].extend(wall_tabs)

    # Apply accumulated operations once per face.
    final_shapes: dict[int, Polygon] = {}
    for fid, base in shapes.items():
        shape = base
        if add_ops[fid]:
            add_union = unary_union(add_ops[fid])
            if not add_union.is_empty:
                shape = shape.union(add_union)
        if sub_ops[fid]:
            sub_union = unary_union(sub_ops[fid])
            if not sub_union.is_empty:
                shape = shape.difference(sub_union)
        final_shapes[fid] = shape

    modified: dict[int, list[tuple[float, float]]] = {}
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] = {}
    for fid, shape in final_shapes.items():
        outer, inners = _shapely_to_vertices(shape)
        modified[fid] = outer
        slot_cutouts[fid] = inners

    return modified, slot_cutouts
