"""CadQuery-backed finger joint generation.

This module keeps the same interval/ownership contract as the shapely engine:
- one seam owner defines finger/slot intervals once
- mate cutouts are mapped from the same owner intervals

Geometry application is done via CadQuery/OCC booleans on planar faces, then
converted back to 2D rings for export.
"""

from __future__ import annotations

from typing import Iterable

import cadquery as cq
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from .face_classifier import SharedEdge
from .finger_joints import (
    DEFAULT_MIN_PLATEAU_LENGTH,
    DEFAULT_NOTCH_BUFFER,
    DEFAULT_PLATEAU_INSET,
    FUSION_DEFAULT_EDGE_MARGIN,
    TAB_DIRECTION_INWARD,
    TAB_DIRECTION_OUTWARD,
    FusionJointParams,
    _build_exclusion_zones,
    _build_fusion_intervals_for_segments,
    _clip_intervals_to_terminal_margins,
    _complement_notch_intervals,
    _corner_endpoint_keepouts,
    _corner_keepouts_near_points,
    _dist_2d,
    _edges_reversed,
    _find_bottom_edge_endpoints,
    _find_edge_index_by_endpoints,
    _find_matching_edge_index,
    _find_plateau_segments,
    _intersect_segment_lists,
    _make_comb_from_intervals,
    _map_intervals_by_param,
    _outward_direction,
    _polygon_to_shapely,
    _reverse_segments,
    _shapely_to_vertices,
)
from .projector import Projection2D, _ordered_vertices_from_edges, _project_point
from .step_loader import PlanarFace, _extract_edges_from_wire


def _is_close_2d(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-6) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _normalize_ring(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ring: list[tuple[float, float]] = []
    for p in points:
        pt = (float(p[0]), float(p[1]))
        if not ring or not _is_close_2d(ring[-1], pt):
            ring.append(pt)

    if len(ring) >= 2 and _is_close_2d(ring[0], ring[-1]):
        ring.pop()
    return ring


def _wire_from_ring_2d(points: list[tuple[float, float]]) -> cq.Wire | None:
    ring = _normalize_ring(points)
    if len(ring) < 3:
        return None
    verts3d = [(x, y, 0.0) for x, y in ring]
    return cq.Wire.makePolygon(verts3d, close=True)


def _face_from_rings_2d(
    outer: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
) -> cq.Face | None:
    outer_wire = _wire_from_ring_2d(outer)
    if outer_wire is None:
        return None

    inner_wires: list[cq.Wire] = []
    for hole in holes or []:
        iw = _wire_from_ring_2d(hole)
        if iw is not None:
            inner_wires.append(iw)

    try:
        return cq.Face.makeFromWires(outer_wire, inner_wires)
    except Exception:
        return None


def _cq_face_from_shapely_polygon(poly: Polygon) -> cq.Face | None:
    if poly.is_empty:
        return None

    geom = poly
    if isinstance(geom, MultiPolygon):
        if not geom.geoms:
            return None
        geom = max(geom.geoms, key=lambda g: g.area)
    elif isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return None
        geom = max(polys, key=lambda g: g.area)

    outer = [(float(x), float(y)) for x, y in geom.exterior.coords[:-1]]
    holes = [[(float(x), float(y)) for x, y in ring.coords[:-1]] for ring in geom.interiors]
    return _face_from_rings_2d(outer, holes)


def _largest_face(shape: cq.Shape) -> cq.Face | None:
    if isinstance(shape, cq.Face):
        return shape
    try:
        faces = list(shape.Faces())
    except Exception:
        return None
    if not faces:
        return None
    return max(faces, key=lambda f: f.Area())


def _shape_to_vertices_2d(shape: cq.Shape) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    face = _largest_face(shape)
    if face is None:
        return [], []

    outer_edges = _extract_edges_from_wire(face.outerWire().wrapped, curve_deflection=0.5)
    outer3d = _ordered_vertices_from_edges(outer_edges, tol=0.5)
    outer = [(x, y) for x, y, _ in outer3d]

    inners: list[list[tuple[float, float]]] = []
    for wire in face.innerWires():
        inner_edges = _extract_edges_from_wire(wire.wrapped, curve_deflection=0.5)
        inner3d = _ordered_vertices_from_edges(inner_edges, tol=0.5)
        ring = [(x, y) for x, y, _ in inner3d]
        if len(ring) >= 3:
            inners.append(ring)

    return outer, inners


def _apply_ops_to_face_cadquery(
    outer: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]],
    add_polys: list[Polygon],
    sub_polys: list[Polygon],
    fallback_base: Polygon,
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    base_face = _face_from_rings_2d(outer, holes)
    if base_face is None:
        return _shapely_to_vertices(fallback_base)

    try:
        shape: cq.Shape = base_face

        add_faces = [f for p in add_polys if (f := _cq_face_from_shapely_polygon(p)) is not None]
        if add_faces:
            shape = shape.fuse(*add_faces)

        sub_faces = [f for p in sub_polys if (f := _cq_face_from_shapely_polygon(p)) is not None]
        if sub_faces:
            shape = shape.cut(*sub_faces)

        shape = shape.clean()
        outer2d, inner2d = _shape_to_vertices_2d(shape)
        if outer2d:
            return outer2d, inner2d
    except Exception:
        pass

    # Per-face fallback: keep behavior deterministic even if OCC fails.
    shape = fallback_base
    if add_polys:
        add_union = unary_union(add_polys)
        if not add_union.is_empty:
            shape = shape.union(add_union)
    if sub_polys:
        sub_union = unary_union(sub_polys)
        if not sub_union.is_empty:
            shape = shape.difference(sub_union)
    return _shapely_to_vertices(shape)


def apply_finger_joints_fusion_cadquery(
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
    """Apply finger joints using owner-contract intervals + CadQuery booleans."""
    if fusion_params is None:
        fusion_params = FusionJointParams()

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
    for fid, proj in projections.items():
        raw_shapes[fid] = _polygon_to_shapely(proj.outer_polygon, proj.inner_polygons)

    exclusion_zones: dict[int, list[Polygon]] = {}
    for fid, proj in projections.items():
        exclusion_zones[fid] = _build_exclusion_zones(proj, notch_buffer)

    add_ops: dict[int, list[Polygon]] = {fid: [] for fid in projections}
    sub_ops: dict[int, list[Polygon]] = {fid: [] for fid in projections}

    # Direct shared-edge joints.
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

        outward_pos = _outward_direction(pos_p1, pos_p2, raw_shapes[pos_id])
        if tab_direction == TAB_DIRECTION_INWARD:
            inward_pos = (-outward_pos[0], -outward_pos[1])
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
            pos_tabs = _make_comb_from_intervals(
                pos_p1, pos_p2, pos_depth, outward_pos, pos_finger_intervals,
                exclusion_zones=exclusion_zones.get(pos_id, []),
            )
            add_ops[pos_id].extend(pos_tabs)

        if neg_depth > 1e-9:
            outward_neg = _outward_direction(neg_p1, neg_p2, raw_shapes[neg_id])
            inward_neg = (-outward_neg[0], -outward_neg[1])
            neg_slots = _make_comb_from_intervals(
                neg_p1, neg_p2, neg_depth, inward_neg, neg_slot_intervals,
                exclusion_zones=exclusion_zones.get(neg_id, []),
            )
            sub_ops[neg_id].extend(neg_slots)

    # Through-slot joints for walls not directly adjacent to bottom.
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

                outward_bottom = _outward_direction(slot_start, slot_end, raw_shapes[bottom_id])
                inward_bottom = (-outward_bottom[0], -outward_bottom[1])
                bottom_slots = _make_comb_from_intervals(
                    slot_start, slot_end, slot_depth, inward_bottom, slot_intervals,
                    exclusion_zones=exclusion_zones.get(bottom_id, []),
                )
                sub_ops[bottom_id].extend(bottom_slots)

                if tab_depth <= 1e-9:
                    continue
                outward_wall = _outward_direction(wall_start, wall_end, raw_shapes[fid])
                if tab_direction == TAB_DIRECTION_INWARD:
                    inward_wall = (-outward_wall[0], -outward_wall[1])
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

    modified: dict[int, list[tuple[float, float]]] = {}
    slot_cutouts: dict[int, list[list[tuple[float, float]]]] = {}
    for fid, proj in projections.items():
        outer2d, inner2d = _apply_ops_to_face_cadquery(
            outer=proj.outer_polygon,
            holes=proj.inner_polygons,
            add_polys=add_ops[fid],
            sub_polys=sub_ops[fid],
            fallback_base=raw_shapes[fid],
        )
        modified[fid] = outer2d
        slot_cutouts[fid] = inner2d

    return modified, slot_cutouts

