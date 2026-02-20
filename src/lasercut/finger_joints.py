"""Finger joint pattern generation using Shapely boolean operations.

Instead of manually inserting vertices into polygon arrays, we use Shapely's
robust boolean operations (union/difference) to add tabs and cut slots.
This correctly handles corners, edge transitions, and complex polygon shapes.
"""

import math
from dataclasses import dataclass
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union
from shapely import affinity

from .projector import Projection2D, _project_point, _normalize, _cross, _sub, _dot
from .face_classifier import SharedEdge
from .step_loader import PlanarFace, EdgeData

DEFAULT_FINGER_WIDTH = 20.0
# Margin at each end of an edge where no fingers are placed.
# Prevents finger clashes at corners where two edges meet.
DEFAULT_EDGE_MARGIN = 10.0
# Buffer distance around inner features (notches/holes) where no fingers are placed.
DEFAULT_NOTCH_BUFFER = 2.0
# Inset from plateau (mountain) boundaries where fingers start/end.
DEFAULT_PLATEAU_INSET = 3.0
# Minimum usable plateau segment length in mm.
DEFAULT_MIN_PLATEAU_LENGTH = 12.0
# Geometric cleanup radius (mm) for optional morphological open/close.
# Keep at 0 by default to preserve original contours (especially rounded corners).
DEFAULT_CLEANUP_RADIUS = 0.0


def _dist_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _lerp_2d(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


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


def _build_tooth_intervals(
    edge_len: float,
    finger_width: float,
    margin: float,
    start_with_tooth: bool,
    plateau_segments: list[tuple[float, float]] | None,
    min_plateau_length: float,
    force_odd_segments: bool = False,
) -> list[tuple[float, float]]:
    """Build parametric tooth intervals once, so both mating faces stay synchronized."""
    if edge_len < 1e-6:
        return []

    margin_t = margin / edge_len if margin > 0 else 0.0
    if plateau_segments:
        segments = []
        for seg_s, seg_e in plateau_segments:
            eff_s = max(seg_s, margin_t)
            eff_e = min(seg_e, 1.0 - margin_t)
            if eff_e > eff_s + 1e-6:
                segments.append((eff_s, eff_e))
    else:
        segments = [(margin_t, 1.0 - margin_t)]

    intervals: list[tuple[float, float]] = []
    start_idx = 0 if start_with_tooth else 1
    for seg_s, seg_e in segments:
        seg_len = (seg_e - seg_s) * edge_len
        if plateau_segments:
            min_len = max(finger_width * 0.5, min_plateau_length)
        else:
            min_len = finger_width * 0.5
        if seg_len < min_len:
            continue
        n = _compute_n_fingers(seg_len, finger_width, margin=0)
        if force_odd_segments and n % 2 == 0:
            n += 1
        for i in range(start_idx, n, 2):
            t0 = seg_s + (i / n) * (seg_e - seg_s)
            t1 = seg_s + ((i + 1) / n) * (seg_e - seg_s)
            if t1 > t0 + 1e-9:
                intervals.append((t0, t1))
    return intervals


def _polygon_to_shapely(outer: list[tuple[float, float]],
                        holes: list[list[tuple[float, float]]] | None = None) -> Polygon:
    """Convert vertex lists to a Shapely Polygon."""
    if len(outer) < 3:
        return Polygon()
    hole_rings = [h for h in (holes or []) if len(h) >= 3]
    return Polygon(outer, hole_rings)


def _simplify_ring(coords: list[tuple[float, float]], tol: float = 0.15) -> list[tuple[float, float]]:
    """Remove degenerate micro-vertices that are closer than *tol* mm apart.

    Preserves polygon topology by keeping every vertex whose removal would
    collapse an edge to less than *tol*.
    """
    if len(coords) < 4:
        return list(coords)
    out: list[tuple[float, float]] = [coords[0]]
    for pt in coords[1:]:
        if _dist_2d(out[-1], pt) >= tol:
            out.append(pt)
    # Close the loop check: first vs last
    if len(out) > 1 and _dist_2d(out[-1], out[0]) < tol:
        out.pop()
    return out if len(out) >= 3 else list(coords)


def _shapely_to_vertices(geom) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """Convert a Shapely geometry back to outer + inner vertex lists."""
    if geom.is_empty:
        return [], []

    if isinstance(geom, MultiPolygon):
        # Take the largest polygon
        geom = max(geom.geoms, key=lambda g: g.area)

    outer = _simplify_ring(list(geom.exterior.coords[:-1]))
    inners = [_simplify_ring(list(ring.coords[:-1])) for ring in geom.interiors]
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


def _make_comb(p1: tuple[float, float], p2: tuple[float, float],
               depth: float, finger_width: float,
               outward_dir: tuple[float, float],
               margin: float = 0.0,
               overlap: float = 0.01,
               exclusion_zones: list[Polygon] | None = None,
               cut_margins: bool = False,
               start_with_tooth: bool = True,
               tooth_intervals: list[tuple[float, float]] | None = None,
               plateau_segments: list[tuple[float, float]] | None = None,
               min_plateau_length: float = 0.0) -> list[Polygon]:
    """Create a list of rectangular 'teeth' along an edge.

    When plateau_segments is provided (non-empty list), teeth are distributed
    independently within each plateau segment. When not provided or empty,
    teeth are distributed across the full usable range (within margin).

    Args:
        p1, p2: Edge endpoints.
        depth: How far the teeth extend (= material thickness).
        finger_width: Target finger width in mm.
        outward_dir: Unit vector pointing in the direction the teeth extend.
        margin: Dead zone at each end of the edge (no fingers placed here).
        overlap: Small overlap to ensure Shapely boolean ops work.
        exclusion_zones: Optional list of Shapely polygons; any tooth that
                        intersects an exclusion zone is skipped.
        cut_margins: If True, also create teeth covering the margin zones at
                    both ends. Used for negative faces (slots) so margin areas
                    are cut flush instead of forming oversized tabs.
        plateau_segments: Optional list of (t_start, t_end) parametric ranges
                         where fingers should be placed. Each segment gets its
                         own evenly-distributed set of fingers.
        min_plateau_length: Minimum segment length (mm) for plateau segments.
                           Segments shorter than this are ignored.

    Returns:
        List of Shapely Polygon rectangles (the teeth).
    """
    teeth = []
    nx, ny = outward_dir
    edge_len = _dist_2d(p1, p2)

    if edge_len < 1e-6:
        return teeth

    margin_t = margin / edge_len if margin > 0 else 0.0

    # Optionally add margin-covering teeth (for negative faces).
    # Extend beyond edge endpoints to cover any ledge artifacts that were
    # cleaned by morphological opening but still referenced by original edges.
    if cut_margins and margin > 0:
        dx = (p2[0] - p1[0]) / edge_len
        dy = (p2[1] - p1[1]) / edge_len
        ext = depth
        p1_ext = (p1[0] - dx * ext, p1[1] - dy * ext)
        p2_ext = (p2[0] + dx * ext, p2[1] + dy * ext)
        for sa, sb in [(p1_ext, _lerp_2d(p1, p2, margin_t)),
                        (_lerp_2d(p1, p2, 1.0 - margin_t), p2_ext)]:
            rect = Polygon([
                (sa[0] - nx * overlap, sa[1] - ny * overlap),
                (sb[0] - nx * overlap, sb[1] - ny * overlap),
                (sb[0] + nx * depth, sb[1] + ny * depth),
                (sa[0] + nx * depth, sa[1] + ny * depth),
            ])
            if rect.is_valid and rect.area > 0:
                teeth.append(rect)

    if tooth_intervals is None:
        tooth_intervals = _build_tooth_intervals(
            edge_len=edge_len,
            finger_width=finger_width,
            margin=margin,
            start_with_tooth=start_with_tooth,
            plateau_segments=plateau_segments,
            min_plateau_length=min_plateau_length,
            force_odd_segments=False,
        )

    zone_union = unary_union(exclusion_zones) if exclusion_zones else None

    for t0, t1 in tooth_intervals:
        s0 = _lerp_2d(p1, p2, t0)
        s1 = _lerp_2d(p1, p2, t1)

        rect = Polygon([
            (s0[0] - nx * overlap, s0[1] - ny * overlap),
            (s1[0] - nx * overlap, s1[1] - ny * overlap),
            (s1[0] + nx * depth, s1[1] + ny * depth),
            (s0[0] + nx * depth, s0[1] + ny * depth),
        ])
        if not rect.is_valid or rect.area <= 0:
            continue

        if zone_union is not None:
            blocked = rect.intersection(zone_union)
            if not blocked.is_empty and blocked.area >= rect.area * 0.5:
                continue

        teeth.append(rect)

    return teeth


def _make_strip(p1: tuple[float, float], p2: tuple[float, float],
                depth: float,
                outward_dir: tuple[float, float],
                margin: float = 0.0,
                overlap: float = 0.01,
                exclusion_zones: list[Polygon] | None = None,
                plateau_segments: list[tuple[float, float]] | None = None,
                min_plateau_length: float = 0.0) -> list[Polygon]:
    """Create full-width strip rectangles along an edge segment set.

    Used to inset a positive-joint edge inward by `depth` before adding tabs
    back, so overall outer dimensions are preserved.
    """
    strips = []
    nx, ny = outward_dir
    edge_len = _dist_2d(p1, p2)
    if edge_len < 1e-6:
        return strips

    margin_t = margin / edge_len if margin > 0 else 0.0
    if plateau_segments:
        segments = []
        for seg_s, seg_e in plateau_segments:
            eff_s = max(seg_s, margin_t)
            eff_e = min(seg_e, 1.0 - margin_t)
            if eff_e > eff_s + 1e-6:
                segments.append((eff_s, eff_e))
    else:
        segments = [(margin_t, 1.0 - margin_t)]

    for seg_s, seg_e in segments:
        seg_len = (seg_e - seg_s) * edge_len
        min_len = max(min_plateau_length, 1e-3)
        if seg_len < min_len:
            continue

        s0 = _lerp_2d(p1, p2, seg_s)
        s1 = _lerp_2d(p1, p2, seg_e)
        rect = Polygon([
            (s0[0] - nx * overlap, s0[1] - ny * overlap),
            (s1[0] - nx * overlap, s1[1] - ny * overlap),
            (s1[0] + nx * depth, s1[1] + ny * depth),
            (s0[0] + nx * depth, s0[1] + ny * depth),
        ])
        if not rect.is_valid or rect.area <= 0:
            continue
        if exclusion_zones and any(rect.intersects(z) for z in exclusion_zones):
            continue
        strips.append(rect)

    return strips


def _filter_tooth_intervals_by_zones(
    p1: tuple[float, float],
    p2: tuple[float, float],
    depth: float,
    outward_dir: tuple[float, float],
    tooth_intervals: list[tuple[float, float]],
    exclusion_zones: list[Polygon] | None,
    overlap: float = 0.01,
) -> list[tuple[float, float]]:
    """Filter tooth intervals whose rectangles intersect exclusion zones.

    Keeps tab/slot interval phasing symmetric while avoiding cuts that would
    intrude into protected neighboring seam regions.
    """
    if not tooth_intervals or not exclusion_zones:
        return tooth_intervals

    nx, ny = outward_dir
    kept: list[tuple[float, float]] = []
    for t0, t1 in tooth_intervals:
        s0 = _lerp_2d(p1, p2, t0)
        s1 = _lerp_2d(p1, p2, t1)
        rect = Polygon([
            (s0[0] - nx * overlap, s0[1] - ny * overlap),
            (s1[0] - nx * overlap, s1[1] - ny * overlap),
            (s1[0] + nx * depth, s1[1] + ny * depth),
            (s0[0] + nx * depth, s0[1] + ny * depth),
        ])
        if not rect.is_valid or rect.area <= 0:
            continue
        if any(rect.intersects(z) for z in exclusion_zones):
            continue
        kept.append((t0, t1))
    return kept


def _filter_tooth_intervals_by_dual_zones(
    p1_a: tuple[float, float],
    p2_a: tuple[float, float],
    depth_a: float,
    outward_a: tuple[float, float],
    zones_a: list[Polygon] | None,
    p1_b: tuple[float, float],
    p2_b: tuple[float, float],
    depth_b: float,
    outward_b: tuple[float, float],
    zones_b: list[Polygon] | None,
    tooth_intervals: list[tuple[float, float]],
    overlap: float = 0.01,
) -> list[tuple[float, float]]:
    """Filter intervals that are valid on both mating faces.

    Used for through-slot joints so we preserve tab/slot complement while also
    keeping the pattern away from protected seam regions on both faces.
    """
    if not tooth_intervals:
        return tooth_intervals

    zones_a = zones_a or []
    zones_b = zones_b or []
    if not zones_a and not zones_b:
        return tooth_intervals

    nax, nay = outward_a
    nbx, nby = outward_b
    kept: list[tuple[float, float]] = []
    for t0, t1 in tooth_intervals:
        sa0 = _lerp_2d(p1_a, p2_a, t0)
        sa1 = _lerp_2d(p1_a, p2_a, t1)
        sb0 = _lerp_2d(p1_b, p2_b, t0)
        sb1 = _lerp_2d(p1_b, p2_b, t1)

        rect_a = Polygon([
            (sa0[0] - nax * overlap, sa0[1] - nay * overlap),
            (sa1[0] - nax * overlap, sa1[1] - nay * overlap),
            (sa1[0] + nax * depth_a, sa1[1] + nay * depth_a),
            (sa0[0] + nax * depth_a, sa0[1] + nay * depth_a),
        ])
        rect_b = Polygon([
            (sb0[0] - nbx * overlap, sb0[1] - nby * overlap),
            (sb1[0] - nbx * overlap, sb1[1] - nby * overlap),
            (sb1[0] + nbx * depth_b, sb1[1] + nby * depth_b),
            (sb0[0] + nbx * depth_b, sb0[1] + nby * depth_b),
        ])
        if (not rect_a.is_valid or rect_a.area <= 0 or
                not rect_b.is_valid or rect_b.area <= 0):
            continue
        if zones_a and any(rect_a.intersects(z) for z in zones_a):
            continue
        if zones_b and any(rect_b.intersects(z) for z in zones_b):
            continue
        kept.append((t0, t1))

    return kept


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

    # Robust side test: for a boundary edge, one side is inside the polygon
    # and the opposite side is outside. This avoids centroid-heuristic errors
    # on concave/irregular outlines.
    probe = min(1.0, max(0.2, length * 0.01))
    test1 = Point(mid[0] + n1[0] * probe, mid[1] + n1[1] * probe)
    test2 = Point(mid[0] + n2[0] * probe, mid[1] + n2[1] * probe)
    inside1 = polygon.buffer(1e-9).contains(test1)
    inside2 = polygon.buffer(1e-9).contains(test2)
    if inside1 and not inside2:
        return n2
    if inside2 and not inside1:
        return n1

    # Fallback when probing is ambiguous (near corners / noisy geometry).
    centroid = polygon.centroid
    d1 = (test1.x - centroid.x) ** 2 + (test1.y - centroid.y) ** 2
    d2 = (test2.x - centroid.x) ** 2 + (test2.y - centroid.y) ** 2
    return n1 if d1 > d2 else n2


def _compute_n_fingers(edge_len: float, finger_width: float, margin: float = 0.0) -> int:
    """Compute number of finger segments (>= 2).

    Uses the usable length (edge_len - 2*margin) for finger count calculation.
    """
    usable = edge_len - 2 * margin
    if usable <= 0:
        return 2
    n = max(1, round(usable / finger_width))
    return max(2, n)


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


def apply_finger_joints(
    projections: dict[int, Projection2D],
    shared_edges: list[SharedEdge],
    bottom_id: int,
    thickness: float,
    finger_width: float = 0,
    kerf: float = 0,
    edge_margin: float = -1,
    notch_buffer: float = -1,
    plateau_inset: float = -1,
    min_plateau_length: float = -1,
    faces: list[PlanarFace] | None = None,
    all_shared_edges: list[SharedEdge] | None = None,
    preserve_outer_dims: bool = True,
    cleanup_radius: float = DEFAULT_CLEANUP_RADIUS,
) -> tuple[dict[int, list[tuple[float, float]]], dict[int, list[list[tuple[float, float]]]]]:
    """Apply finger joints using Shapely boolean operations.

    For each shared edge:
    - Detects plateau (mountain) segments on both faces
    - Intersects plateau segments so tabs and slots match
    - Skips tiny plateau segments below min_plateau_length
    - Distributes fingers evenly within each plateau segment
    - Positive face: union with comb (tabs protrude outward)
    - Negative face: difference with comb (slots cut inward)

    Convention:
    - Bottom is always "positive" (tabs protrude outward)
    - For wall-to-wall edges, the face with the lower ID is positive

    Returns:
        (face_id -> polygon vertices, face_id -> list of inner polygon cutouts)
    """
    if finger_width <= 0:
        finger_width = DEFAULT_FINGER_WIDTH
    if edge_margin < 0:
        edge_margin = DEFAULT_EDGE_MARGIN
    if notch_buffer < 0:
        notch_buffer = DEFAULT_NOTCH_BUFFER
    if plateau_inset < 0:
        plateau_inset = DEFAULT_PLATEAU_INSET
    if min_plateau_length < 0:
        min_plateau_length = DEFAULT_MIN_PLATEAU_LENGTH

    kerf_half = kerf / 2

    # Convert all projections to Shapely Polygons
    # Keep raw shapes (before morphological cleaning) for plateau detection,
    # since the cleaning can smooth out notch features we need to detect.
    raw_shapes: dict[int, Polygon] = {}
    shapes: dict[int, Polygon] = {}
    for fid, proj in projections.items():
        shape = _polygon_to_shapely(proj.outer_polygon, proj.inner_polygons)
        raw_shapes[fid] = shape
        # Optional, conservative morphological cleanup.
        # Disabled by default to preserve true outer contours/radii.
        if cleanup_radius > 1e-6:
            if fid == bottom_id:
                # Bottom plate: closing (fills tiny inward artifacts)
                closed = shape.buffer(cleanup_radius, join_style='round').buffer(-cleanup_radius, join_style='round')
            else:
                # Walls: opening (removes tiny outward artifacts)
                closed = shape.buffer(-cleanup_radius, join_style='round').buffer(cleanup_radius, join_style='round')
            if not closed.is_empty and closed.is_valid:
                if isinstance(closed, MultiPolygon):
                    closed = max(closed.geoms, key=lambda g: g.area)
                shape = closed
        shapes[fid] = shape

    # Build exclusion zones from inner polygons (notches/holes) for each face
    exclusion_zones: dict[int, list[Polygon]] = {}
    for fid, proj in projections.items():
        exclusion_zones[fid] = _build_exclusion_zones(proj, notch_buffer)

    # Faces directly adjacent to the bottom. Used to orient wall-to-wall joints
    # so non-bottom-adjacent walls (e.g. stepped/notched back walls) are expanded
    # toward the box interior while bottom-adjacent side walls are cut inward.
    bottom_adjacent: set[int] = set()
    for se in shared_edges:
        if se.face_a_id == bottom_id:
            bottom_adjacent.add(se.face_b_id)
        elif se.face_b_id == bottom_id:
            bottom_adjacent.add(se.face_a_id)

    # Apply finger joints along shared edges
    for se in shared_edges:
        fid_a = se.face_a_id
        fid_b = se.face_b_id

        if fid_a not in projections or fid_b not in projections:
            continue

        # Determine positive/negative
        # Keep historical orientation for matching wall-to-wall patterns.
        # Outer-dimension preservation is handled separately via strip-inset logic.
        if fid_a == bottom_id:
            pos_id, neg_id = fid_a, fid_b
        elif fid_b == bottom_id:
            pos_id, neg_id = fid_b, fid_a
        else:
            # For wall-to-wall joints, if one wall is bottom-adjacent and the
            # other is not, keep the non-bottom-adjacent wall positive. This is
            # the original pairing convention and yields matching edge phasing.
            a_adj = fid_a in bottom_adjacent
            b_adj = fid_b in bottom_adjacent
            if a_adj != b_adj:
                pos_id = fid_a if not a_adj else fid_b
                neg_id = fid_b if pos_id == fid_a else fid_a
            else:
                pos_id = min(fid_a, fid_b)
                neg_id = max(fid_a, fid_b)

        # Keep outer-dimension preservation focused on bottom-wall joints.
        # Applying the inset on generic wall-wall edges can interfere with
        # later through-slot tab generation on those same walls.
        is_bottom_pair = (fid_a == bottom_id or fid_b == bottom_id)
        preserve_on_this_pair = preserve_outer_dims and is_bottom_pair
        # Start shared joints on a tooth interval so the wall-side profile
        # begins with a slot near corners, reducing visible endpoint ledges
        # in unfolded seam previews.
        start_with_tooth = True

        # Find the matching 2D edges on both faces
        pos_proj = projections[pos_id]
        neg_proj = projections[neg_id]
        pos_edge_idx = _find_matching_edge_index(pos_proj, se)
        neg_edge_idx = _find_matching_edge_index(neg_proj, se)

        if pos_edge_idx is None or neg_edge_idx is None:
            continue

        pos_p1, pos_p2 = pos_proj.outer_edges_2d[pos_edge_idx]
        neg_p1, neg_p2 = neg_proj.outer_edges_2d[neg_edge_idx]

        pos_edge_len = _dist_2d(pos_p1, pos_p2)
        neg_edge_len = _dist_2d(neg_p1, neg_p2)
        if pos_edge_len < 1e-6 or neg_edge_len < 1e-6:
            continue

        # Detect plateau segments on both faces using raw (pre-morphological)
        # shapes, since morphological opening can smooth out notch features.
        pos_plateaus = _find_plateau_segments(pos_p1, pos_p2, raw_shapes[pos_id],
                                              plateau_inset=plateau_inset)
        neg_plateaus = _find_plateau_segments(neg_p1, neg_p2, raw_shapes[neg_id],
                                              plateau_inset=plateau_inset)

        # Determine shared plateau segments (so tabs and slots align)
        # If both faces have no notches (empty plateaus), use default full-edge
        reversed_edge = _edges_reversed(pos_proj, pos_edge_idx, neg_proj, neg_edge_idx)
        if pos_plateaus and neg_plateaus:
            # Both have notches: intersect, accounting for possible edge reversal
            neg_in_pos_space = _reverse_segments(neg_plateaus) if reversed_edge else neg_plateaus
            shared_pos = _intersect_segment_lists(pos_plateaus, neg_in_pos_space)
            shared_neg = _reverse_segments(shared_pos) if reversed_edge else shared_pos
        elif pos_plateaus:
            # Only positive face has notches: use its plateaus for both
            shared_pos = pos_plateaus
            shared_neg = _reverse_segments(pos_plateaus) if reversed_edge else pos_plateaus
        elif neg_plateaus:
            # Only negative face has notches: use its plateaus for both
            shared_neg = neg_plateaus
            shared_pos = _reverse_segments(neg_plateaus) if reversed_edge else neg_plateaus
        else:
            # No notches on either face: use default full-edge distribution
            shared_pos = []
            shared_neg = []

        # Build tooth intervals once in parametric edge space and reuse them
        # on both faces (mirrored when edge directions are opposite).
        tooth_pos = _build_tooth_intervals(
            edge_len=pos_edge_len,
            finger_width=finger_width,
            margin=edge_margin,
            start_with_tooth=start_with_tooth,
            plateau_segments=shared_pos,
            min_plateau_length=min_plateau_length,
            force_odd_segments=True,
        )
        tooth_neg = _reverse_segments(tooth_pos) if reversed_edge else list(tooth_pos)

        # Process both faces
        for face_id, is_positive, p1, p2, plateaus in [
            (pos_id, True, pos_p1, pos_p2, shared_pos),
            (neg_id, False, neg_p1, neg_p2, shared_neg),
        ]:
            outward = _outward_direction(p1, p2, shapes[face_id])
            depth = thickness
            if depth <= 0:
                continue

            # Keep paired joints strictly complementary: do not apply per-face
            # exclusion filtering here, otherwise one side may drop a tooth
            # while the mating side keeps it.
            zones: list[Polygon] = []

            if is_positive:
                if preserve_on_this_pair:
                    # Inset the edge inward by one thickness and then add tabs
                    # back from the inset line to the original line. This keeps
                    # the outer contour envelope unchanged.
                    inward = (-outward[0], -outward[1])
                    strips = _make_strip(
                        p1, p2, depth, inward,
                        # Inset across the full edge to avoid corner step lips
                        # at margin boundaries where adjacent seams meet.
                        margin=0.0,
                        exclusion_zones=zones,
                        plateau_segments=plateaus,
                        min_plateau_length=min_plateau_length,
                    )
                    if strips:
                        shapes[face_id] = shapes[face_id].difference(unary_union(strips))
                    sp1 = (p1[0] + inward[0] * depth, p1[1] + inward[1] * depth)
                    sp2 = (p2[0] + inward[0] * depth, p2[1] + inward[1] * depth)
                    margin_fills: list[Polygon] = []
                    edge_len = _dist_2d(p1, p2)
                    if edge_margin > 0 and edge_len > 1e-6:
                        margin_t = min(0.5, edge_margin / edge_len)
                        # Fill dead-zones at both seam ends back to the original
                        # contour so preserve-mode does not leave L-shaped lips.
                        for t0, t1 in ((0.0, margin_t), (1.0 - margin_t, 1.0)):
                            if t1 <= t0 + 1e-9:
                                continue
                            s0 = _lerp_2d(sp1, sp2, t0)
                            s1 = _lerp_2d(sp1, sp2, t1)
                            fill = Polygon([
                                (s0[0], s0[1]),
                                (s1[0], s1[1]),
                                (s1[0] + outward[0] * depth, s1[1] + outward[1] * depth),
                                (s0[0] + outward[0] * depth, s0[1] + outward[1] * depth),
                            ])
                            if fill.is_valid and fill.area > 0:
                                margin_fills.append(fill)
                    teeth = _make_comb(
                        sp1, sp2, depth, finger_width, outward,
                        margin=edge_margin,
                        exclusion_zones=zones,
                        start_with_tooth=start_with_tooth,
                        tooth_intervals=tooth_pos,
                        plateau_segments=plateaus,
                        min_plateau_length=min_plateau_length,
                    )
                    if margin_fills:
                        teeth.extend(margin_fills)
                else:
                    teeth = _make_comb(p1, p2, depth, finger_width, outward,
                                       margin=edge_margin, exclusion_zones=zones,
                                       start_with_tooth=start_with_tooth,
                                       tooth_intervals=tooth_pos,
                                       plateau_segments=plateaus,
                                       min_plateau_length=min_plateau_length)
            else:
                inward = (-outward[0], -outward[1])
                teeth = _make_comb(p1, p2, depth, finger_width, inward,
                                   margin=edge_margin, exclusion_zones=zones,
                                   cut_margins=False,
                                   start_with_tooth=start_with_tooth,
                                   tooth_intervals=tooth_neg,
                                   plateau_segments=plateaus,
                                   min_plateau_length=min_plateau_length)

            if not teeth:
                continue

            comb = unary_union(teeth)

            if is_positive:
                shapes[face_id] = shapes[face_id].union(comb)
            else:
                shapes[face_id] = shapes[face_id].difference(comb)

    # Apply through-slot joints for walls not adjacent to bottom
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

            bottom_proj = projections[bottom_id]
            protected_bottom_joint_zones: list[Polygon] = []
            protect_dist = max(0.5, notch_buffer) + max(0.0, thickness)
            # Keep through-slot cuts away from direct bottom-wall seam edges.
            for se in shared_edges:
                if se.face_a_id == bottom_id:
                    other = se.face_b_id
                elif se.face_b_id == bottom_id:
                    other = se.face_a_id
                else:
                    continue
                if other not in bottom_adjacent:
                    continue
                edge_idx = _find_matching_edge_index(bottom_proj, se)
                if edge_idx is None:
                    continue
                ep1, ep2 = bottom_proj.outer_edges_2d[edge_idx]
                zone = LineString([ep1, ep2]).buffer(protect_dist)
                if not zone.is_empty:
                    protected_bottom_joint_zones.append(zone)

            # Protect wall-to-wall seam edges from through-slot tab additions.
            protected_wall_joint_zones: dict[int, list[Polygon]] = {}
            for fid, proj in projections.items():
                if fid == bottom_id:
                    continue
                zones: list[Polygon] = []
                for se in shared_edges:
                    if se.face_a_id == fid:
                        other = se.face_b_id
                    elif se.face_b_id == fid:
                        other = se.face_a_id
                    else:
                        continue
                    # Through-slot shaping must not disturb bottom-facing seams
                    # on neighboring walls.
                    if other == bottom_id or other not in projections:
                        continue
                    edge_idx = _find_matching_edge_index(proj, se)
                    if edge_idx is None:
                        continue
                    ep1, ep2 = proj.outer_edges_2d[edge_idx]
                    zone = LineString([ep1, ep2]).buffer(protect_dist)
                    if not zone.is_empty:
                        zones.append(zone)
                protected_wall_joint_zones[fid] = zones

            # Through-slot tabs/slots must stay symmetric as a pair.
            bottom_zones: list[Polygon] = protected_bottom_joint_zones

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

                # --- Through-slots on the bottom plate ---
                slot_start = _project_point(p_start_3d, bottom_proj.origin_3d,
                                            bottom_proj.u_axis, bottom_proj.v_axis)
                slot_end = _project_point(p_end_3d, bottom_proj.origin_3d,
                                          bottom_proj.u_axis, bottom_proj.v_axis)
                slot_len = _dist_2d(slot_start, slot_end)
                if slot_len < 1e-6:
                    continue

                inward = _outward_direction(slot_start, slot_end, shapes[bottom_id])
                inward = (-inward[0], -inward[1])

                slot_depth = thickness

                # Detect plateaus on the wall face for this edge
                wall_proj = projections[fid]
                wall_start = _project_point(p_start_3d, wall_proj.origin_3d,
                                            wall_proj.u_axis, wall_proj.v_axis)
                wall_end = _project_point(p_end_3d, wall_proj.origin_3d,
                                          wall_proj.u_axis, wall_proj.v_axis)
                wall_edge_len = _dist_2d(wall_start, wall_end)
                if wall_edge_len < 1e-6:
                    continue

                wall_plateaus = _find_plateau_segments(wall_start, wall_end, raw_shapes[fid],
                                                       plateau_inset=plateau_inset)
                bottom_plateaus = _find_plateau_segments(slot_start, slot_end, raw_shapes[bottom_id],
                                                         plateau_inset=plateau_inset)

                # Prefer the wall's plateau map for through-slots.
                # The bottom face can encode complementary valleys for the same
                # geometric pattern, which would make a strict intersection empty
                # and incorrectly trigger full-edge finger placement.
                if wall_plateaus and bottom_plateaus:
                    intersection = _intersect_segment_lists(wall_plateaus, bottom_plateaus)
                    # Ignore degenerate overlaps that are too short to host even
                    # one usable finger span, then fall back to wall plateaus.
                    min_useful = max(finger_width * 0.5, min_plateau_length)
                    intersection = [
                        (s, e) for s, e in intersection
                        if (e - s) * wall_edge_len >= min_useful - 1e-6
                    ]
                    shared_plateaus = intersection if intersection else wall_plateaus
                elif wall_plateaus:
                    shared_plateaus = wall_plateaus
                elif bottom_plateaus:
                    shared_plateaus = bottom_plateaus
                else:
                    shared_plateaus = []

                shared_tooth_intervals = _build_tooth_intervals(
                    edge_len=slot_len,
                    finger_width=finger_width,
                    margin=edge_margin,
                    start_with_tooth=False,
                    plateau_segments=shared_plateaus,
                    min_plateau_length=min_plateau_length,
                    force_odd_segments=True,
                )
                # Keep through-slot tooth phasing unfragmented. Filtering these
                # intervals by protection zones can create tiny seam artifacts
                # at interval boundaries.

                slot_teeth = _make_comb(slot_start, slot_end, slot_depth, finger_width, inward,
                                        margin=edge_margin, exclusion_zones=None,
                                        tooth_intervals=shared_tooth_intervals,
                                        plateau_segments=shared_plateaus,
                                        min_plateau_length=min_plateau_length)
                if slot_teeth:
                    slot_comb = unary_union(slot_teeth)
                    shapes[bottom_id] = shapes[bottom_id].difference(slot_comb)

                # Also cut outward from the through-slot line toward the polygon
                # edge.  This turns internal-only holes into visible edge notches,
                # making the through-slot positions clear in the unfolded SVG.
                outward_bottom = (-inward[0], -inward[1])
                edge_notch_teeth = _make_comb(
                    slot_start, slot_end, thickness * 3, finger_width,
                    outward_bottom,
                    margin=edge_margin, exclusion_zones=None,
                    tooth_intervals=shared_tooth_intervals,
                    plateau_segments=shared_plateaus,
                    min_plateau_length=min_plateau_length)
                if edge_notch_teeth:
                    edge_notch_comb = unary_union(edge_notch_teeth)
                    shapes[bottom_id] = shapes[bottom_id].difference(edge_notch_comb)

                # --- Tabs on the wall ---
                wall_zones: list[Polygon] = []
                outward_wall = _outward_direction(wall_start, wall_end, shapes[fid])
                wall_zones = protected_wall_joint_zones.get(fid, [])

                tab_depth = thickness
                if tab_depth <= 0:
                    continue

                # Keep through-slot tab intervals synchronized with bottom slots.
                # Use exclusion zones at geometry stage (below) instead of
                # dropping parametric intervals asymmetrically.

                # Through-slot walls must keep real protruding tabs so they can
                # enter bottom slots. Do not inset/flush them for outer-dim mode.
                wall_teeth = _make_comb(wall_start, wall_end, tab_depth, finger_width,
                                        outward_wall,
                                        margin=edge_margin, exclusion_zones=wall_zones,
                                        tooth_intervals=shared_tooth_intervals,
                                        plateau_segments=shared_plateaus,
                                        min_plateau_length=min_plateau_length)
                if wall_teeth:
                    wall_comb = unary_union(wall_teeth)
                    shapes[fid] = shapes[fid].union(wall_comb)

    # Apply kerf compensation as contour offset:
    # - Positive kerf expands exterior boundaries and contracts holes,
    #   yielding nominal cut dimensions after material removal.
    # - Negative kerf does the inverse (sometimes used for looser fits).
    if abs(kerf_half) > 1e-9:
        for fid, shape in list(shapes.items()):
            compensated = shape.buffer(kerf_half, join_style='mitre')
            if compensated.is_empty:
                continue
            if isinstance(compensated, MultiPolygon):
                compensated = max(compensated.geoms, key=lambda g: g.area)
            if compensated.is_valid and not compensated.is_empty:
                shapes[fid] = compensated

    # Convert Shapely geometries back to vertex lists
    modified: dict[int, list[tuple[float, float]]] = {}
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] = {}

    for fid, shape in shapes.items():
        outer, inners = _shapely_to_vertices(shape)
        modified[fid] = outer

        # Separate inner polygons: original face holes vs new slot cutouts
        # The original inner polygons from the projection are already in the Shapely result
        slot_cutouts[fid] = inners

    return modified, slot_cutouts
